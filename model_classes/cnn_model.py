import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAudioClassifier(nn.Module):
    def __init__(self, num_classes, dropout):
        super(CNNAudioClassifier, self).__init__()
        
        # Layer 1: convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Layer 2: convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Layer 3: convolutional layer
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Dropout layer: helps prevent overfitting by randomly dropping units during training
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers: for classification after the convolutional layers
        self.fc1 = nn.Linear(64 * 96, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Forward pass through the first convolutional layer, followed by ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # Forward pass through the second convolutional layer, followed by ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # Forward pass through the third convolutional layer, followed by ReLU activation and max pooling
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply dropout and pass through the first fully connected layer with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)     # Output layer, no activation here
        
        return x
