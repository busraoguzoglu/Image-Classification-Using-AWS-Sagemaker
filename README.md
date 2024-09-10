# Dog Breed Classification using PyTorch and AWS SageMaker

This project focuses on developing a deep learning model to classify dog breeds using PyTorch, leveraging AWS SageMaker for training, hyperparameter tuning, and deployment. The model is based on the ResNet18 architecture, a popular convolutional neural network designed for image classification tasks.

## Data
The dataset used for this project consists of images of 133 different dog breeds. The dataset is organized into three directories: train, valid, and test. Each subdirectory contains images categorized by breed. The data is stored in an AWS S3 bucket and is loaded directly into SageMaker for training.


## Model Training
The model is based on the ResNet18 architecture, which is fine-tuned on our dataset using transfer learning. The training process is managed by train_model_profiling.py, which includes steps for data preprocessing, model training, and evaluation.


## Hyperparameter Tuning
Hyperparameter optimization is performed using SageMaker's HyperparameterTuner. The script hpo.py defines the search space and objective metrics for tuning. SageMaker automatically manages the tuning job, allowing for efficient exploration of hyperparameter combinations to improve model performance.

**Hyperparameters Tuned:**
  
- Learning rate
- Batch size

## Debugging and Profiling
To ensure optimal model performance and detect potential issues during training, debugging and profiling were integrated into the project using AWS SageMaker Debugger. The DebuggerHookConfig and ProfilerConfig settings help monitor the training process by capturing data on the model's gradients, loss, and resource utilization.

**Debugging:** The model's gradients and loss values are tracked to detect problems like vanishing gradients, overfitting, and poor weight initialization. SageMaker Debugger's built-in rules are applied to automatically detect and flag such issues.

**Profiling:** Profiling is used to monitor system resource utilization, such as CPU and memory usage, throughout the training process. This helps identify bottlenecks and optimize resource allocation to improve training efficiency.

## Deployment and Inference

Due to the nature of PyTorch models, a custom inference script (inference.py) was required to handle the model loading and prediction logic. This script manages how input data (images in this case) is processed and passed through the model for predictions.

The model is deployed on an instance on AWS that serves the endpoint, allowing us to send images of dogs and receive predictions about their breed. A custom predictor class is used to handle image data and ensure it is properly serialized and deserialized for the prediction requests. This setup ensures efficient deployment and accurate predictions for the dog breed classification task.
