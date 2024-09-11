"""
Script to perform training, profiling, debugging
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse

from PIL import ImageFile
import torchvision.datasets as datasets
from torchvision.models import resnet18
import s3fs
import os
from torch.utils.data import Dataset, DataLoader
from sagemaker.debugger import DebuggerHookConfig, CollectionConfig, Rule, rule_configs
import logging
import smdebug.pytorch as smd
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, hook):   
    
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Turn off gradients for validation/testing
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = running_loss / len(test_loader)
    
    logger.info(f"Average Loss: {average_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy}")
    
    print(f'Test set: Average loss: {average_loss:.4f}, Test Accuracy: {accuracy}%')



        
def train(model, train_loader, criterion, optimizer, epochs, hook):
    
    hook.set_mode(smd.modes.TRAIN)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    return model




def net():
    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)  # Use the pretrained=True argument to load pre-trained weights
    
    for param in model.parameters():
        param.requires_grad = False  # Freeze all the layers

    # Replace the final fully connected layer to match the number of classes (133 dog breeds)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 133)  # Update the output layer to match the number of classes
    
    return model


def create_data_loaders(s3_data_dir, batch_size):
    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=False)

    # Temporary local directory to simulate the structure
    local_data_dir = '/tmp/dogImages'  
    if not os.path.exists(local_data_dir):
        os.makedirs(local_data_dir)

    # Create local directories to mimic S3 structure
    for subset in ['train', 'valid', 'test']:
        s3_subset_path = f"{s3_data_dir}/{subset}"
        local_subset_path = os.path.join(local_data_dir, subset)
        if not os.path.exists(local_subset_path):
            os.makedirs(local_subset_path)
        
        # List all class directories in the S3 subset directory
        class_dirs = fs.ls(s3_subset_path)
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            local_class_dir = os.path.join(local_subset_path, class_name)
            if not os.path.exists(local_class_dir):
                os.makedirs(local_class_dir)
            
            # List all image files in the S3 class directory
            s3_files = fs.ls(class_dir)
            
            for file_path in s3_files:
                if file_path.endswith(('jpg', 'jpeg', 'png')):  # Consider only image files
                    local_file_path = os.path.join(local_class_dir, os.path.basename(file_path))
                    if not os.path.exists(local_file_path):
                        fs.get(file_path, local_file_path)  # Download the file locally

    # Define transforms for training, validation, and testing sets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
        transforms.ToTensor(),  # Convert images to tensor format
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet's mean and std
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(local_data_dir, 'train'), transform=transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(local_data_dir, 'valid'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(local_data_dir, 'test'), transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def main(args):

    model=net()
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(args.s3_data_dir, args.batch_size)
    
    hook = smd.Hook.create_from_json_file()
    
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    logger.info("Hooks created")
    
    logger.info("Model training")
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, hook)
    
    test(model, test_loader, loss_criterion, hook)
    
    # Ensure the output directory exists
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Save the model
    print(f"Saving model to {os.path.join(args.model_dir, 'model.pth')}")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--s3_data_dir', type=str, default="sagemaker-studio-814424935677-pve38pywqph/dogImages", help='S3 directory where data is stored')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--data_dir', type=str, default='./dogImages', help='Directory where the dataset is stored')
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'], help='Directory where the model will be saved')
    
    args = parser.parse_args()

    main(args)