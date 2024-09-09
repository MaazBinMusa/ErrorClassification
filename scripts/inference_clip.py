model_name = "clip_text_error_similarity_model"
processor_name = "clip_text_error_similarity_processor"


import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms
import io
import pickle as pkl
import pandas as pd

def load_data(path):

    with open(path, "rb") as f:
        df = pkl.load(f)
        
    return df

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


df = load_data("data/induced_errors_v1.pkl")

# Load the fine-tuned CLIP model from the Hugging Face Hub
model = CLIPModel.from_pretrained("maazmusa/clip_text_error_similarity_model")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Example image and text (replace with your actual image and text)
image = preprocess_image(df.iloc[0]["cropped_image"]["bytes"]) 
text = df.iloc[0]["answer"]

# Preprocess the input using the processor
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

# Move inputs to the GPU if available
input_ids = inputs['input_ids'].to(device)
pixel_values = inputs['pixel_values'].to(device)

# Perform inference
with torch.no_grad():
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    predicted_similarity = outputs.logits_per_image.sigmoid()  # Normalize with sigmoid if trained that way

# Print the predicted similarity score (between 0 and 1)
print(f"Predicted Similarity: {predicted_similarity.item():.4f}")
print(f"Actual Similarity: {df.iloc[0]['distance']:.4f}")
print("Text:", text)
print("Old Text:", df.iloc[0]["old_text"])
