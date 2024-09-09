import torch
import pandas as pd
from tqdm import tqdm
from typing import Union
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import io
import json
import pickle
import torch.nn as nn
from torchvision import models as vismodels
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = image_transform(image)

    return image_tensor.unsqueeze(0)

def image_to_vector(image_bytes, model):
    
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
    model.eval()
    model.cuda()
    
    # Convert image bytes to tensor and preprocess
    image_tensor = preprocess_image(image_bytes).cuda()
    
    with torch.no_grad():
        return model(image_tensor).squeeze().cpu()

def text_to_vector(text, tokenizer, model):
    
    model.eval()
    model.cuda()
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()


with open("data/induced_errors.pkl","rb") as file:
    df = pickle.load(file)
    
df = df.fillna("") \
    .replace("None", "") \
    .replace("nan", "") \
    .replace("null", "")
    
train_df, test_df = split_df(df)

# replace all non negative labels with 1
train_df["label"] = train_df["label"].apply(lambda x: 1 if x >= 0 else 0)
test_df["label"] = test_df["label"].apply(lambda x: 1 if x >= 0 else 0)

# replace all -1 labels with 0
train_df["label"] = train_df["label"].apply(lambda x: 0 if x == -1 else x)
test_df["label"] = test_df["label"].apply(lambda x: 0 if x == -1 else x)

print(df.columns)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
vision_model = vismodels.resnet18(pretrained=True)

x_train, y_train, z_train = [],[],[]
x_tests, y_tests, z_tests = [],[],[]

maxid = 400000

# convert to tensors
id = 0
for index,row in train_df.iterrows():
    
    comb_vector = []
    comb_vector.extend(text_to_vector(row["answer"],bert_tokenizer,bert_model))
    #comb_vector.extend(image_to_vector(row["cropped_image"]["bytes"],vision_model)) 
    x_train.append(comb_vector)
    y_train.append(row["label"])
    z_train.append(row["error_name"])
    if id == maxid:
        break
    id += 1
    

# convert to tensors
id = 0
for index,row in test_df.iterrows():
    comb_vector = []
    comb_vector.extend(text_to_vector(row["answer"],bert_tokenizer,bert_model))
    #comb_vector.extend(image_to_vector(row["cropped_image"]["bytes"],vision_model)) 
    x_tests.append(comb_vector)
    y_tests.append(row["label"])
    z_tests.append(row["error_name"])
    if id == maxid:
        break
    id += 1
    
# Logistic Regression
print("Fitting Logistic Regression")
lr_model = LogisticRegression(multi_class='ovr', max_iter=3000)
lr_model.fit(x_train, y_train)

lr_accuracy = lr_model.score(x_tests, y_tests)
pred = lr_model.predict(x_tests)
actu = y_tests

with open("data/lr-results.json","w") as file:
    json.dump(
        {
            "pred":pred,
            "actual":actu,
            "error":z_tests
        },
        file,
        indent=4
    )
# print(f"LR_ACCURACY: {lr_accuracy}")

# # save all things
# with open("data/LR/train_df.pkl","wb") as file:
#     pickle.dump(train_df,file)
# with open("data/LR/test_df.pkl","wb") as file:
#     pickle.dump(test_df,file)
# with open("data/LR/x_train.pkl","wb") as file:
#     pickle.dump(x_train,file)
# with open("data/LR/y_train.pkl","wb") as file:
#     pickle.dump(y_train,file)
# with open("data/LR/z_train.pkl","wb") as file:
#     pickle.dump(z_train,file)
# with open("data/LR/x_test.pkl","wb") as file:
#     pickle.dump(x_tests,file)
# with open("data/LR/y_test.pkl","wb") as file:
#     pickle.dump(y_tests,file)
# with open("data/LR/z_test.pkl","wb") as file:
#     pickle.dump(z_tests,file)
# with open("data/LR/lr_model.pkl","wb") as file:
#     pickle.dump(lr_model,file)
