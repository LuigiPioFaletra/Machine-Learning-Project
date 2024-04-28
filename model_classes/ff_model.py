import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_layers: list,
        num_classes: int,
        dropout: float = 0.2
    ):
        super(FFNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        
        self.layers = torch.nn.ModuleList()
        
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_layers):
            layer = nn.Linear(prev_size, hidden_size)
            activation = nn.ReLU()
            dropout_layer = nn.Dropout(p=dropout)
            self.layers.extend([layer, activation, dropout_layer])
            prev_size = hidden_size
            
        self.output_layer = nn.Linear(prev_size, num_classes)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        return x