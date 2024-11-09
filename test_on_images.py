import os
import torch
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

# Paths for input directory and output file
input_dir = "/home/shruti/shruti_bt/all_datasets/datasets/coralclassification/CoralClass.v1i.multiclass/valid"  # Replace with your source folder path
output_file = "/home/shruti/shruti_bt/all_datasets/datasets/yuxiang/valid.txt"  # Single output text file

# Open the output file in write mode
with open(output_file, "w") as f:
    # Process each image in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        
        # Ensure the file is an image
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Predict label
            predicted_label = predict(img_path)
            
            # Write result to the single output file
            f.write(f"{img_name}: {predicted_label}\n")
            print(f"Processed {img_name}: Predicted class - {predicted_label}")
