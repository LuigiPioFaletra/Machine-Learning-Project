import torch
import torch.nn as nn

class FFAudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout):
        super(FFAudioClassifier, self).__init__()
        
        # Layer 1: fully connected layers from 768 to 384 features
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])

        # Layer 2: fully connected layers from 384 to 256 features
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])

        # Layer 3: fully connected layers from 256 to 192 features
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])

        # Layer 4: fully connected layers from 192 to 128 features
        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])

        # Layer 5: fully connected layers from 128 to 96 features
        self.fc5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])

        # Layer 6: fully connected layers from 96 to 64 features
        self.fc6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])

        # Layer 7: fully connected layers from 64 to 48 features
        self.fc7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.bn7 = nn.BatchNorm1d(hidden_layers[6])

        # Layer 8: fully connected layers from 48 to 32 features
        self.fc8 = nn.Linear(hidden_layers[6], hidden_layers[7])
        self.bn8 = nn.BatchNorm1d(hidden_layers[7])

        # Layer 9: fully connected layers from 32 to 24 features
        self.fc9 = nn.Linear(hidden_layers[7], hidden_layers[8])
        self.bn9 = nn.BatchNorm1d(hidden_layers[8])

        # Layer 10: fully connected layers from 24 to 16 features
        self.fc10 = nn.Linear(hidden_layers[8], hidden_layers[9])
        self.bn10 = nn.BatchNorm1d(hidden_layers[9])

        # Layer 11: fully connected layers from 16 to 8 features
        self.fc11 = nn.Linear(hidden_layers[9], num_classes)

        # ELU activation function: provides fast convergence and robust learning
        self.elu = nn.ELU()

        # Dropout layer: helps prevent overfitting by randomly dropping units during training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Generates a tensor of random values from a normal distribution and adds randomness to the data
        x = x + torch.randn_like(x) * 0.1
        
        # First fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # Second fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # Third fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Fourth fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Fifth fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Sixth fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # Seventh fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc7(x)
        x = self.bn7(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # Eighth fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc8(x)
        x = self.bn8(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Nineth fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc9(x)
        x = self.bn9(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Tenth fully connected layer + batch normalization + ELU activation + dropout
        x = self.fc10(x)
        x = self.bn10(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # Output layer, no activation here
        x = self.fc11(x)
        
        return x
