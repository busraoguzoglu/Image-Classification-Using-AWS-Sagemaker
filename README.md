**Dog Breed Classification using PyTorch and AWS SageMaker**

This project focuses on developing a deep learning model to classify dog breeds using PyTorch, leveraging AWS SageMaker for training, hyperparameter tuning, and deployment. The model is based on the ResNet18 architecture, a popular convolutional neural network designed for image classification tasks.

*Project Structure*

├── dogImages/                    # Directory containing the dataset

│   ├── train/                    # Training images

│   ├── valid/                    # Validation images

│   └── test/                     # Test images

├── train_model.py                # Script for training the model

├── hpo.py                        # Script for hyperparameter optimization

├── train_and_deploy.ipynb        # Jupyter notebook for training, tuning, and deployment

├── README.md                     # Project README file


*Data*
The dataset used for this project consists of images of 133 different dog breeds. The dataset is organized into three directories: train, valid, and test. Each subdirectory contains images categorized by breed. The data is stored in an AWS S3 bucket and is loaded directly into SageMaker for training.


*Model Training*
The model is based on the ResNet18 architecture, which is fine-tuned on our dataset using transfer learning. The training process is managed by train_model.py, which includes steps for data preprocessing, model training, and evaluation.


*Hyperparameter Tuning*
Hyperparameter optimization is performed using SageMaker's HyperparameterTuner. The script hpo.py defines the search space and objective metrics for tuning. SageMaker automatically manages the tuning job, allowing for efficient exploration of hyperparameter combinations to improve model performance.


- Hyperparameters Tuned:
  
- Learning rate
  
- Batch size (optional)
