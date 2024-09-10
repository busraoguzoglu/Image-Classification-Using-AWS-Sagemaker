import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


# Code adapted from https://github.com/jdboachie/image-classification-using-aws-sagemaker/blob/main/inference.py

def Net():
    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)  # Use the pretrained=True argument to load pre-trained weights
    
    for param in model.parameters():
        param.requires_grad = False  # Freeze all the layers

    # Replace the final fully connected layer to match the number of classes (133 dog breeds)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 133)  # Update the output layer to match the number of classes
    
    return model



def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model




# input_fn: for handling incoming requests and deserializing them
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')

    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))  # Load image from bytes
    
    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']  # Extract URL from JSON
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))  # Load image from bytes
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    
    
    

# predict_fn: for performing inference using the model and preprocessed input
def predict_fn(input_object, model):
    logger.info('In predict fn')

    # Define image transformation pipeline
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet's mean and std
    ])
    
    logger.info("Transforming input image")
    input_object = test_transform(input_object)  # Apply transformations
    input_object = input_object.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    with torch.no_grad():  # No gradient calculation needed during inference
        logger.info("Calling model for prediction")
        prediction = model(input_object)  # Run the prediction
    
    return prediction