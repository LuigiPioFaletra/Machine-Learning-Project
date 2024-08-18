import torch
import torch.nn as nn

class CNNAudioClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, stride, padding, num_classes):
        super(CNNAudioClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=padding)                           # First convolutional layer
        self.relu = nn.ReLU()                                                                                               # ReLU activation function
        self.bn1 = nn.BatchNorm1d(out_channels)                                                                             # First batch normalization

        self.conv2 = nn.Conv1d(out_channels, int(out_channels * 1.5), kernel_size, stride=2, padding=padding)               # Second convolutional layer
        self.bn2 = nn.BatchNorm1d(int(out_channels * 1.5))                                                                  # Second batch normalization

        self.conv3 = nn.Conv1d(int(out_channels * 1.5), int(out_channels * 2), kernel_size, stride=2, padding=padding)      # Third convolutional layer
        self.bn3 = nn.BatchNorm1d(int(out_channels * 2))                                                                    # Third batch normalization
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)        # Global average pooling
        self.flatten = nn.Flatten()                           # Flattening layer
        self.fc1 = nn.Linear(int(out_channels * 2), 128)      # First fully connected layer
        self.dropout = nn.Dropout(p=0.3)                      # Dropout layer for regularization
        self.fc2 = nn.Linear(128, num_classes)                # Second fully connected layer

    def forward(self, x):
        # Pass the input through the first convolutional block (conv1 -> ReLU -> bn1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

      # Pass the input through the first convolutional block (conv2 -> ReLU -> bn2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        # Pass the input through the first convolutional block (conv3 -> ReLU -> bn3)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)

        x = self.global_avg_pool(x)     # Apply global average pooling to reduce the feature map dimensions
        x = x.view(x.size(0), -1)       # Flatten the tensor to prepare it for the fully connected layers

        # Pass through the first fully connected layer, followed by ReLU and dropout for regularization
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass through the second fully connected layer to produce the final output logits for classification
        x = self.fc2(x)

        return x
