import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from transformers import ViTImageProcessor
from model import ViTForImageClassification
import pandas as pd
from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# Custom Dataset class
class CoralDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # Assuming first column is the image file name
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1]  # Assuming second column is the class label
        if self.transform:
            image = self.transform(image)
        return image, label

# Paths
train_img_dir = '/home/shruti/shruti_bt/all_datasets/datasets/coralclassification/Coral Classification.v1i.multiclass/train/'
train_csv_file = os.path.join(train_img_dir, '_classes.csv')

valid_img_dir = '/home/shruti/shruti_bt/all_datasets/datasets/coralclassification/Coral Classification.v1i.multiclass/valid/'
valid_csv_file = os.path.join(valid_img_dir, '_classes.csv')

test_img_dir = '/home/shruti/shruti_bt/all_datasets/datasets/coralclassification/Coral Classification.v1i.multiclass/test/'
test_csv_file = os.path.join(test_img_dir, '_classes.csv')

# Dataset and DataLoaders
train_ds = CoralDataset(train_img_dir, train_csv_file, transform=ToTensor())
valid_ds = CoralDataset(valid_img_dir, valid_csv_file, transform=ToTensor())
test_ds = CoralDataset(test_img_dir, test_csv_file, transform=ToTensor())
# print("Number of train samples: ", len(train_ds))
# print("Number of test samples: ", len(test_ds))


# Model parameters
EPOCHS = 25
BATCH_SIZE = 1
LEARNING_RATE = 2e-5

# Define Model
model = ViTForImageClassification(num_labels=2)  # 2 classes: bleached and healthy
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()

# Use GPU if available  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = model.to(device)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# Directory to save checkpoints
checkpoint_dir = "/home/shruti/shruti_bt/classification_model/checkpoints2"
log_file_path = "/home/shruti/shruti_bt/classification_model/training_log2.txt"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
for epoch in range(EPOCHS):        
    for step, (x, y) in enumerate(train_loader):
        x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
        for index, array in enumerate(x):
            x[index] = np.squeeze(array)
        x = torch.tensor(np.stack(feature_extractor(x, do_rescale=False)['pixel_values'], axis=0)).to(device)
        y = y.to(device)

        # Forward pass
        output, loss = model(x, None)
        if loss is None: 
            loss = loss_func(output, y)   
            optimizer.zero_grad()           
            loss.backward()                 
            optimizer.step()

        if step % 50 == 0:
            # Test accuracy on the test set
            test = next(iter(test_loader))
            test_x = np.split(np.squeeze(np.array(test[0])), BATCH_SIZE)
            for index, array in enumerate(test_x):
                test_x[index] = np.squeeze(array)
            test_x = torch.tensor(np.stack(feature_extractor(test_x, do_rescale=False)['pixel_values'], axis=0)).to(device)
            test_y = test[1].to(device)

            test_output, _ = model(test_x, test_y)
            test_output = test_output.argmax(1)
            accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
            log_message = f"Epoch: {epoch} | Train Loss: {loss:.4f} | Test Accuracy: {accuracy:.2f}"
            print(log_message)
            
            # Write to log file
            with open(log_file_path, "a") as f:
                f.write(log_message + "\n")

    torch.save(model, os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt"))
    print(f"Checkpoint saved")






# EVAL_BATCH = 1
# valid_loader  = DataLoader(valid_ds, batch_size=EVAL_BATCH, shuffle=True, num_workers=4) 
# # Disable grad
# with torch.no_grad():
    
#   inputs, target = next(iter(valid_loader))
#   # Reshape and get feature matrices as needed
#   print(inputs.shape)
#   inputs = inputs[0].permute(1, 2, 0)
#   # Save original Input
#   originalInput = inputs
#   for index, array in enumerate(inputs):
#     inputs[index] = np.squeeze(array)
#   inputs = torch.tensor(np.stack(feature_extractor(inputs, do_rescale=False)['pixel_values'], axis=0))

#   # Send to appropriate computing device
#   inputs = inputs.to(device)
#   target = target.to(device)
 
#   # Generate prediction
#   prediction, loss = model(inputs, target)
    
#   # Predicted class value using argmax
#   predicted_class = np.argmax(prediction.cpu())
#   value_predicted = list(valid_ds.class_to_idx.keys())[list(valid_ds.class_to_idx.values()).index(predicted_class)]
#   value_target = list(valid_ds.class_to_idx.keys())[list(valid_ds.class_to_idx.values()).index(target)]
        
#   # Show result
#   plt.imshow(originalInput)
#   plt.xlim(224,0)
#   plt.ylim(224,0)
#   plt.title(f'Prediction: {value_predicted} - Actual target: {value_target}')
#   plt.show()