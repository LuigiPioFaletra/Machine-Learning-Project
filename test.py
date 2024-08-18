import os
import torch
import torch.nn as nn

from data_classes.fma_dataset import FMADataset
from model_classes.cnn_model import CNNAudioClassifier
from model_classes.ff_model import FFAudioClassifier
from torch.utils.data import DataLoader
from utils import evaluate


if __name__ == '__main__':
    # Configuration parameters
    root = os.path.join(os.path.dirname(__file__), 'data', 'fma_small')                 # Path to directory containing audio files
    metadata_file = os.path.join(os.path.dirname(__file__), 'data', 'tracks.csv')       # Path to CSV file with metadata
    sample_rate = 16000                                                                 # Sample rate for audio data
    max_duration = 30                                                                   # Maximum duration of audio clips in seconds
    batch_size = 8                                                                      # Batch size for DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")               # Use GPU if available, otherwise use CPU
    embedding_dim = 768                                                                 # Dimensionality of embeddings (for FF model)

    # Load the test dataset
    test_ds_cnn = FMADataset(root, metadata_file, 'test', sample_rate, max_duration)    # Test dataset for CNN model
    test_dl_cnn = DataLoader(test_ds_cnn, batch_size=batch_size, shuffle=False)         # DataLoader for CNN test set
    test_ds_ff = FMADataset(root, metadata_file, 'test', sample_rate, max_duration)     # Test dataset for FF model
    test_dl_ff = DataLoader(test_ds_ff, batch_size=batch_size, shuffle=False)           # DataLoader for FF test set

    # Define the model parameters
    in_channels = 1                             # Number of input channels for CNN
    out_channels = 16                           # Number of output channels in the first convolutional layer
    input_length = sample_rate * max_duration   # Length of input audio data
    kernel_size = 3                             # Size of the convolutional kernel
    stride = 1                                  # Stride for the convolutional layers
    padding = 1                                 # Padding for the convolutional layers
    num_classes = len(test_ds_cnn.genres)       # Number of classes, or genres, for classification

    # Instantiate the models
    cnn_model = CNNAudioClassifier(in_channels, out_channels, input_length, kernel_size, stride, padding, num_classes).to(device)
    ff_model = FFAudioClassifier(embedding_dim=embedding_dim, num_classes=num_classes).to(device)

    # Load pre-trained model weights
    cnn_model.load_state_dict(torch.load("best_cnn_audio_classifier.pth"))      # Load weights for CNN model
    print("CNN Model loaded.")
    ff_model.load_state_dict(torch.load("best_ff_audio_classifier.pth"))        # Load weights for FF model
    print("FF Model loaded.")

    criterion = nn.CrossEntropyLoss()                                           # Define loss function
    models = [("CNN", cnn_model, test_dl_cnn), ("FF", ff_model, test_dl_ff)]    # Evaluate both models
    
    for model_name, model, test_dl in models:
        test_metrics = evaluate(model, test_dl, criterion, device)              # Evaluate the model on the test dataset

        # Print evaluation metrics
        for key, value in test_metrics.items():
            print(f"{model_name} Model Test {key}: {value:.4f}")
