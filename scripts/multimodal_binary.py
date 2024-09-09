import io
import torch
import random
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from transformers import CLIPModel
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from src.gen_errors.precision_errors import PrecisionErrors


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, max_length=77):
        self.dataframe = dataframe
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Process text
        text = row['answer']
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process image

        image_bytes = row["cropped_image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.image_transform(image)

        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'pixel_values': image_tensor,
            'label': label,
            'text': text,
            "image_bytes": image_bytes
        }


class ExtendedCLIPClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=2):
        super(ExtendedCLIPClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.text_projection = nn.Linear(self.clip.text_model.config.hidden_size, self.clip.config.projection_dim)
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Process image
        image_features = self.clip.vision_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        image_features = self.clip.visual_projection(image_features)

        # Process text
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        text_features = self.text_projection(text_features)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Combine features
        combined_features = (image_features + text_features) / 2

        # Classify
        logits = self.classifier(combined_features)
        return logits


def train_model(train_df):
    dataset = MultimodalDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=12)

    # Example usage
    model = ExtendedCLIPClassifier().to('cuda')
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            pixel_values = batch['pixel_values'].to('cuda')
            labels = batch['label'].to('cuda')
            texts = batch['text']
            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


def test_model(model, test_df):
    """
    This function will record the actual label [1,0] and the predicted label [1,0] for each test sample.
    :param model:
    :param test_df:
    :return: list of actual labels and predicted labels
    """

    dataset = MultimodalDataset(test_df)
    dataloader = DataLoader(dataset, batch_size=12)

    actual_labels = []
    predicted_labels = []

    model.eval()

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        pixel_values = batch['pixel_values'].to('cuda')
        labels = batch['label'].to('cuda')
        outputs = model(input_ids, attention_mask, pixel_values)
        _, predicted = torch.max(outputs, 1)
        actual_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    return actual_labels, predicted_labels


def split_df(df: pd.DataFrame, train_ratio: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
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


print(df["label"].value_counts())

# break the df into train and test
train_df, test_df = split_df(df)

# replace all non negative labels with 1
train_df["label"] = train_df["label"].apply(lambda x: 1 if x >= 0 else 0)
test_df["label"] = test_df["label"].apply(lambda x: 1 if x >= 0 else 0)

# replace all -1 labels with 0
train_df["label"] = train_df["label"].apply(lambda x: 0 if x == -1 else x)
test_df["label"] = test_df["label"].apply(lambda x: 0 if x == -1 else x)

# train the model
fine_tuned_model = train_model(train_df)

# test the model
actual_labels, predicted_labels = test_model(fine_tuned_model, test_df)

# find accuracy, precision, recall, f1 score
tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(actual_labels)):
    if actual_labels[i] == 1 and predicted_labels[i] == 1:
        tp += 1
    elif actual_labels[i] == 0 and predicted_labels[i] == 1:
        fp += 1
    elif actual_labels[i] == 0 and predicted_labels[i] == 0:
        tn += 1
    elif actual_labels[i] == 1 and predicted_labels[i] == 0:
        fn += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")