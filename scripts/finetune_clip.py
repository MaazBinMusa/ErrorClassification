import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import pandas as pd
from typing import Union
import numpy as np
from torchvision import transforms
from PIL import Image
import io
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_data(path):

    with open(path, "rb") as f:
        df = pkl.load(f)
        
    return df

def split_df(df: pd.DataFrame, train_ratio: float = 0.8) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    This function will split the dataframe into train and test dataframes.
    :param df: Original dataframe
    :param train_ratio: Ratio of train data
    :return: train_df, test_df
    """

    # Get unique labels
    labels = df['label'].unique()

    train_indices = []
    test_indices = []

    # Sample train_ratio of each label for training
    for label in labels:
        label_indices = df[df['label'] == label].index
        n_train = int(len(label_indices) * train_ratio)

        train_label_indices = np.random.choice(label_indices, size=n_train, replace=False)
        test_label_indices = np.setdiff1d(label_indices, train_label_indices)

        train_indices.extend(train_label_indices)
        test_indices.extend(test_label_indices)

    # Create train and test dataframes
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    return train_df, test_df

def preprocess_image(image_bytes):
    
    image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = image_transform(image)
    # the image_tensor right now is [1, 3, 224, 224], we need to remove the first dimension
    image_tensor = image_tensor.squeeze(0)
    
    return image_tensor

# Load the pre-trained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = 'cuda'
clip_model = clip_model.to(device)

# Example Dataset class
class SimilarityDataset(Dataset):
    def __init__(self, texts, images, similarities):
        self.texts = texts
        self.images = images
        self.similarities = similarities
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'image': self.images[idx],
            'similarity': self.similarities[idx]
        }

df = load_data("data/induced_errors_v1.pkl")
df_train, df_test = split_df(df)
max_samples = 2000000000

# Ready the data for training
texts = df_train["answer"].to_list()[:max_samples]
images = [ preprocess_image(item["bytes"]) for item in tqdm(df_train["cropped_image"].to_list()[:max_samples])]
similarities = df_train["distance"].to_list()[:max_samples]

dataset = SimilarityDataset(texts, images, similarities)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the loss function (MSE loss for regression)
loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)

# Fine-tuning the CLIP model for similarity prediction
optimizer = torch.optim.AdamW(clip_model.parameters(), lr=2e-5)

epochs = 20
# Training loop for fine-tuning
for epoch in range(epochs):  # Set your number of epochs
    clip_model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        inputs = processor(text=batch['text'], images=batch['image'], return_tensors="pt", padding=True, do_rescale=False)
        
        # Move inputs to GPU
        input_ids = inputs['input_ids'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        labels = torch.tensor(batch['similarity'], dtype=torch.float32).unsqueeze(1).to(device)  # Labels to GPU
        
        # Get CLIP similarity logits
        outputs = clip_model(input_ids=input_ids, pixel_values=pixel_values)
        
        # CLIP returns similarity logits; we use them to predict the similarity score
        predicted_similarities = outputs.logits_per_image.sigmoid()  # Optionally normalize with sigmoid
        
        # Compute loss between predicted similarities and ground truth similarities
        loss = loss_fn(predicted_similarities, labels)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {epoch_loss}")
    print("Epoch completed")
        

# move model to cpu
clip_model = clip_model.to('cpu')

# upload this model to the hub
clip_model.save_pretrained("clip/clip_text_error_similarity_model", safe_serialization=False, push_to_hub=True)

# save the test df
df_test.to_pickle("data/clip/df_test.pkl")
df_train.to_pickle("data/clip/df_train.pkl")