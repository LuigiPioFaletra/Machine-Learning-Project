import torch
import torch.nn as nn

class CNNAudioClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, stride, padding, num_classes):
        super(CNNAudioClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)         # First convolutional layer
        self.relu = nn.ReLU()                                                                   # ReLU activation function
        self.bn1 = nn.BatchNorm1d(out_channels)                                                 # First Batch Normalization
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)                                       # Max Pooling
        self.conv2 = nn.Conv1d(out_channels, out_channels * 2, kernel_size, stride, padding)    # Second convolutional layer
        self.bn2 = nn.BatchNorm1d(out_channels * 2)                                             # Second Batch Normalization

        # These formulas calculate the length of the output after first and second convolution and pooling
        conv_output_length = (input_length // 2 + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        conv_output_length = (conv_output_length // 2 + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        self.flatten = nn.Flatten()                                                             # Flattening layer
        self.fc1 = nn.Linear(out_channels * 2 * conv_output_length, 256)                        # First fully connected layer
        self.fc2 = nn.Linear(256, num_classes)                                                  # Second fully connected layer
        self.dropout = nn.Dropout(p=0.5)                                                        # Dropout layer for regularization

    def forward(self, x):
        # Pass the input through the first convolutional block (conv1 -> ReLU -> BN1 -> Pool)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool(x)

        # Pass the output through the second convolutional block (conv2 -> ReLU -> BN2 -> Pool)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool(x)

        # Flatten the output to feed it into the fully connected layers
        x = self.flatten(x)

        # Pass through the first fully connected layer, followed by ReLU and dropout for regularization
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass through the second fully connected layer to produce the logits for classification
        x = self.fc2(x)

        return x
