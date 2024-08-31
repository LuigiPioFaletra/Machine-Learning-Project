import torch
import torch.nn as nn
import torch.nn.functional as F

class FFAudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout):
        super(FFAudioClassifier, self).__init__()
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)

        # Dropout Layer: helps prevent overfitting by randomly dropping units during training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # First fully connected layer + ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)         # Apply dropout
        
        # Second fully connected layer + ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)         # Apply dropout
        
        # Third fully connected layer + ReLU activation
        x = F.relu(self.fc3(x))
        x = self.dropout(x)         # Apply dropout
        
        # Output layer
        x = self.fc4(x)
        
        return x
