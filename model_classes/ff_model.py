import torch
import torch.nn as nn

class FFAudioClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(FFAudioClassifier, self).__init__()
        
        self.fc1 = nn.Linear(embedding_dim, 256)            # First fully connected layer
        self.relu = nn.ReLU()                               # ReLU activation function
        self.dropout = nn.Dropout(p=0.1)                    # Dropout layer for regularization
        self.fc2 = nn.Linear(256, num_classes)              # Second fully connected layer
        
    def forward(self, x):
        x = x.view(x.size(0), -1)       # Flatten the tensor from (batch_size, channels, length) to (batch_size, channels * length)
        
        # Pass through the first fully connected layer, followed by ReLU and dropout for regularization
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass through the second fully connected layer to produce the logits for classification
        x = self.fc2(x)

        return x
