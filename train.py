import contextlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_classes.fma_dataset import FMADataset
from model_classes.cnn_model import CNNAudioClassifier
from model_classes.ff_model import FFAudioClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate

# Settings and parameters
root = os.path.join(os.path.dirname(__file__), 'data', 'fma_small')             # Path to directory containing audio files
metadata_file = os.path.join(os.path.dirname(__file__), 'data', 'tracks.csv')   # Path to metadata CSV file
sample_rate = 16000                                                             # Sample rate for audio files
max_duration = 30                                                               # Maximum duration in seconds of audio clips
batch_size = 8                                                                  # Batch size for training and validation
num_epochs = 1                                                                  # Number of epochs for training
learning_rate = 0.001                                                           # Learning rate for the optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # Use GPU if available, otherwise use CPU

# Early stopping parameters
patience = 5                    # Number of epochs with no improvement after which training will be stopped
best_val_loss = float('inf')    # Initialize best validation loss to infinity
best_val_accuracy = 0           # Initialize best validation accuracy to 0
epochs_no_improve = 0           # Counter for epochs with no improvement
early_stop = False              # Flag to indicate if early stopping should be triggered

# Load data
train_ds_cnn = FMADataset(root, metadata_file, 'training', sample_rate, max_duration)   # Training dataset for CNN model
val_ds_cnn = FMADataset(root, metadata_file, 'validation', sample_rate, max_duration)   # Validation dataset for CNN model
train_ds_ff = FMADataset(root, metadata_file, 'training', sample_rate, max_duration)    # Training dataset for FF model
val_ds_ff = FMADataset(root, metadata_file, 'validation', sample_rate, max_duration)    # Validation dataset for FF model

# Create DataLoaders for both models
train_dl_cnn = DataLoader(train_ds_cnn, batch_size=batch_size, shuffle=True)    # DataLoader for training CNN model
val_dl_cnn = DataLoader(val_ds_cnn, batch_size=batch_size, shuffle=False)       # DataLoader for validation CNN model
train_dl_ff = DataLoader(train_ds_ff, batch_size=batch_size, shuffle=True)      # DataLoader for training FF model
val_dl_ff = DataLoader(val_ds_ff, batch_size=batch_size, shuffle=False)         # DataLoader for validation FF model

# Define the model
in_channels = 1                             # Number of input channels for CNN
out_channels = 16                           # Number of output channels for first convolutional layer
input_length = sample_rate * max_duration   # Length of input audio
kernel_size = 3                             # Kernel size for convolutional layers
stride = 1                                  # Stride for convolutional layers
padding = 1                                 # Padding for convolutional layers
num_classes = len(train_ds_cnn.genres)      # Number of output classes, which coincides with the number of genres
embedding_dim = 768                         # Dimensionality of embeddings, that depends on pre-trained model used

# Instantiate models
cnn_model = CNNAudioClassifier(in_channels, out_channels, input_length, kernel_size, stride, padding, num_classes).to(device)
ff_model = FFAudioClassifier(embedding_dim=embedding_dim, num_classes=num_classes).to(device)

# Define loss function and optimizer
loss_fn_cnn = nn.CrossEntropyLoss()                                     # Loss function for CNN model
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=learning_rate)    # Optimizer for CNN model
loss_fn_ff = nn.CrossEntropyLoss()                                      # Loss function for FF model
optimizer_ff = optim.Adam(ff_model.parameters(), lr=learning_rate)      # Optimizer for FF model

accumulation_steps = 4                      # Number of steps for gradient accumulation
use_amp = torch.cuda.is_available()         # Flag to check if Automatic Mixed Precision (AMP) should be used

# Use Automatic Mixed Precision (AMP) if CUDA is available
if use_amp:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()       # Gradient scaler for mixed precision
else:
    from contextlib import nullcontext
    autocast = nullcontext      # No-op context manager to use when AMP is not used
    scaler = None

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, accumulation_steps=4, scaler=None):
    model.train()                                                           # Set the model to training mode
    p_bar = tqdm(dataloader, desc="\nTraining")                             # Progress bar for training
    running_loss = 0.0
    predictions = []
    references = []
    optimizer.zero_grad()                                                   # Zero the gradients of the optimizer
    
    for i, (x, y) in enumerate(p_bar):
        try:
            x, y = x.to(device), y.to(device)                               # Move data to the specified device

            with autocast():                                                # Use automatic mixed precision if enabled
                outputs = model(x)                                          # Forward pass
                loss = loss_fn(outputs, y)                                  # Compute loss

            if scaler:
                scaler.scale(loss).backward()                               # Scale and backward pass if using mixed precision
            else:
                loss.backward()                                             # Regular backward pass
            
            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)                                  # Step the optimizer if using mixed precision
                    scaler.update()                                         # Update the scaler
                else:
                    optimizer.step()                                        # Regular optimizer step
                optimizer.zero_grad()                                       # Zero the gradients for the next step

            running_loss += loss.item()                                     # Accumulate loss
            preds = torch.argmax(outputs, dim=1)                            # Get predicted class labels
            predictions.extend(preds.cpu().numpy())                         # Store predictions
            references.extend(y.cpu().numpy())                              # Store true labels
            p_bar.set_postfix({"loss": running_loss / len(dataloader)})     # Update progress bar with current loss
        
        except Exception as e:
            print(f"An error occurred while processing the batch: {str(e)}")
            continue
    
    # Compute accuracy only if there are references
    if len(references) > 0:
        accuracy = (np.array(references) == np.array(predictions)).mean()
    else:
        accuracy = float('nan')                     # Set accuracy to NaN if there are no references

    return {
        "loss": running_loss / len(dataloader),     # Average loss over the epoch
        "accuracy": accuracy                        # Accuracy of the model
    }

# Early stopping parameters
def init_early_stopping():
    return {
        "best_val_loss": float('inf'),              # Initialize best validation loss to infinity
        "best_val_accuracy": 0,                     # Initialize best validation accuracy
        "epochs_no_improve": 0,                     # Counter for epochs with no improvement
        "early_stop": False                         # Flag to indicate if early stopping should be triggered
    }

# Define models, dataloaders, loss functions, and optimizers
models = [
    {"name": "CNN", "model": cnn_model, "train_dl": train_dl_cnn, "val_dl": val_dl_cnn, "loss_fn": loss_fn_cnn, "optimizer": optimizer_cnn, "early_stopping": init_early_stopping()},
    {"name": "FF", "model": ff_model, "train_dl": train_dl_ff, "val_dl": val_dl_ff, "loss_fn": loss_fn_ff, "optimizer": optimizer_ff, "early_stopping": init_early_stopping()}
]

def check_early_stopping(val_metrics, early_stopping, model, model_name):
    if val_metrics['loss'] < early_stopping["best_val_loss"]:

        # Update best validation loss and accuracy
        early_stopping["best_val_loss"] = val_metrics['loss']
        early_stopping["best_val_accuracy"] = val_metrics['accuracy']
        early_stopping["epochs_no_improve"] = 0
        
        torch.save(model.state_dict(), f"best_{model_name.lower()}_audio_classifier.pth")   # Save the model with the best validation loss
    else:
        early_stopping["epochs_no_improve"] += 1                                            # Increment counter if no improvement

    # Check if early stopping criteria are met
    if early_stopping["epochs_no_improve"] >= patience:
        print(f"No improvement for {patience} epochs in {model_name}, stopping early.")
        early_stopping["early_stop"] = True


if __name__ == '__main__':
    # Training loop with early stopping for both models
    for epoch in range(num_epochs):
        if any(m["early_stopping"]["early_stop"] for m in models):
            print("Early stopping for one or more models")
            break
        
        try:
            for m in models:
                if m["early_stopping"]["early_stop"]:
                    continue                                                                    # Skip training for this model if early stopping was triggered
                
                # Training
                train_metrics = train_one_epoch(m["model"], m["train_dl"], m["loss_fn"], m["optimizer"], device, accumulation_steps, scaler)
                val_metrics = evaluate(m["model"], m["val_dl"], m["loss_fn"], device)
                
                print(f"Epoch {epoch+1}/{num_epochs} - {m['name']} Model")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"Validation Loss: {val_metrics['loss']:.4f}, Validation Accuracy: {val_metrics['accuracy']:.4f}")
                check_early_stopping(val_metrics, m["early_stopping"], m["model"], m["name"])   # Check for early stopping
        
        except Exception as e:
            print(f"An error occurred during epoch {epoch+1}: {str(e)}")
        
        # Clear CUDA cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
