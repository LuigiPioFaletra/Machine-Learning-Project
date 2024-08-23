import torch
import torch.nn as nn

class CNNAudioClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, stride, padding, num_classes):
        super(CNNAudioClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=10, stride=5)     # First convolutional layer
        self.ln1 = nn.LayerNorm(out_channels)                                           # First batch normalization
        self.gelu = nn.GELU()                                                           # GeLU activation function
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2)     # Second convolutional layer
        self.ln2 = nn.LayerNorm(out_channels)                                           # Second batch normalization

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2)     # Third convolutional layer
        self.ln3 = nn.LayerNorm(out_channels)                                           # Third batch normalization

        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2)     # Fourth convolutional layer
        self.ln4 = nn.LayerNorm(out_channels)                                           # Fourth batch normalization

        self.conv5 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2)     # Fifth convolutional layer
        self.ln5 = nn.LayerNorm(out_channels)                                           # Fifth batch normalization

        self.conv6 = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)     # Sixth convolutional layer
        self.ln6 = nn.LayerNorm(out_channels)                                           # Sixth batch normalization

        self.conv7 = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)     # Seventh convolutional layer
        self.ln7 = nn.LayerNorm(out_channels)                                           # Seventh batch normalization
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)      # Global average pooling
        self.flatten = nn.Flatten()                         # Flattening layer
        self.fc1 = nn.Linear(out_channels, 64)              # First fully connected layer
        self.dropout = nn.Dropout(p=0.1)                    # Dropout layer for regularization
        self.fc2 = nn.Linear(64, num_classes)               # Second fully connected layer

    def forward(self, x):
        # Pass the input through the first convolutional block (conv1 -> ln1 -> GeLU)
        x = self.conv1(x)
        x = self.ln1(x.permute(0, 2, 1))                    # Permute to shape [batch, length, channels] for LayerNorm
        x = x.permute(0, 2, 1)                              # Permute back to [batch, channels, length]
        x = self.gelu(x)

        # Pass the input through the second convolutional block (conv2 -> ln2 -> GeLU)
        x = self.conv2(x)
        x = self.ln2(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.gelu(x)

        # Pass the input through the third convolutional block (conv3 -> ln3 -> GeLU)
        x = self.conv3(x)
        x = self.ln3(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.gelu(x)

        # Pass the input through the fourth convolutional block (conv4 -> ln4 -> GeLU)
        x = self.conv4(x)
        x = self.ln4(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.gelu(x)

        # Pass the input through the fifth convolutional block (conv5 -> ln5 -> GeLU)
        x = self.conv5(x)
        x = self.ln5(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.gelu(x)

        # Pass the input through the sixth convolutional block (conv6 -> ln6 -> GeLU)
        x = self.conv6(x)
        x = self.ln6(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.gelu(x)

        # Pass the input through the seventh convolutional block (conv7 -> ln7 -> GeLU)
        x = self.conv7(x)
        x = self.ln7(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.gelu(x)
        
        x = self.global_avg_pool(x)                         # Output shape: [batch_size, out_channels, 1]
        x = x.view(x.size(0), -1)                           # Flatten to shape [batch_size, out_channels]

        # Pass through the first fully connected layer, followed by ReLU and dropout for regularization
        x = self.dropout(x)                                 # Apply dropout and pass through the fully connected layer
        x = self.fc1(x)                                     # Output shape: [batch_size, 64]

        # Pass through the second fully connected layer to produce the final output logits for classification
        x = self.fc2(x)                                     # Output shape: [64, num_classes]

        return x
