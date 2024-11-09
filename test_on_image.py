import torch
import numpy as np
from transformers import ViTImageProcessor
from PIL import Image

# Path to a saved model checkpoint
checkpoint_path = "/home/shruti/shruti_bt/classification_model/checkpoints2/model_epoch_25.pt"

# Load model and set to evaluation mode
model = torch.load(checkpoint_path)
model.eval()

# Load feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def preprocess_image(img_path):
    # Open image and apply transformations
    image = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")['pixel_values']
    return inputs.to(device)

def predict(img_path):
    # Preprocess the image
    inputs = preprocess_image(img_path)
    
    # Inference
    with torch.no_grad():
        outputs, _ = model(inputs, None)  # Forward pass
        _, predicted_class = torch.max(outputs, 1)  # Get predicted class
        
    # Map output to class names
    class_labels = {0: "Bleached", 1: "Healthy"}
    predicted_label = class_labels[predicted_class.item()]
    return predicted_label

# Example usage
test_image_path = "/home/shruti/shruti_bt/all_datasets/datasets/yuxiang/extractedframesoutput/01.png"  # Replace with your test image path
predicted_label = predict(test_image_path)
print(f"Predicted class: {predicted_label}")


