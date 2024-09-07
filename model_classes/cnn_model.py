import torch
import torch.nn as nn

class CNNAudioClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, dropout):
        super(CNNAudioClassifier, self).__init__()

        # Layer 1: convolutional layer with 8 filters
        self.conv1 = nn.Conv1d(input_channels, 8, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(8)

        # Layer 2: convolutional layer with 16 filters
        self.conv2 = nn.Conv1d(8, 16, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)

        # Layer 3: convolutional layer with 32 filters
        self.conv3 = nn.Conv1d(16, 32, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        # Layer 4: convolutional layer with 64 filters
        self.conv4 = nn.Conv1d(32, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(64)

        # Layer 5: convolutional layer with 128 filters
        self.conv5 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(128)

        # GELU activation function: provides a good balance between sparsity and smoothness
        self.gelu = nn.GELU()

        # Dropout layer: helps prevent overfitting by randomly dropping units during training
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers: for classification after the convolutional layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)
    
    def forward(self, x):
        # Adds a new dimension at index 1 and swaps dimensions 1 and 2 of the tensor
        x = x.unsqueeze(1).transpose(1, 2)

        # Generates a tensor of random values from a normal distribution and adds randomness to the data
        x = x + torch.randn_like(x) * 0.1

        # Forward pass through the first convolutional layer + batch normalization + GELU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        # Forward pass through the second convolutional layer + batch normalization + GELU activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        # Forward pass through the third convolutional layer + batch normalization + GELU activation
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)

        # Forward pass through the fourth convolutional layer + batch normalization + GELU activation
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu(x)

        # Forward pass through the fifth convolutional layer + batch normalization + GELU activation
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.gelu(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # First fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        
        # Second fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gelu(x)
        
        # Third fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.gelu(x)
        
        # Output layer, no activation here
        x = self.fc4(x)
        
        return x
