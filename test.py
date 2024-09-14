import joblib
import numpy as np
import os
import torch
import torch.nn as nn

from addict import Dict
from model_classes.cnn_model import CNNAudioClassifier
from model_classes.ff_model import FFAudioClassifier
from sklearn.metrics import hinge_loss
from torch.utils.data import DataLoader, TensorDataset
from train import load_or_extract_data
from utils import compute_metrics, evaluate
from yaml_config_override import add_arguments

if __name__ == '__main__':
    config = Dict(add_arguments())      # Load configuration from a YAML file or command-line arguments
    device = torch.device('cuda' if torch.cuda.is_available() and config.training.device == 'cuda' else 'cpu')

    # Check if test embeddings and labels already exist and upload them, otherwise extract and save them
    test_emb, test_lab = load_or_extract_data(config.data.dataset_dir,
                                              config.data.metadata_file,
                                              config.training.sample_rate,
                                              config.training.max_duration,
                                              config.training.batch_size,
                                              config.training.device,
                                              config.training.npy_dir,
                                              config.split.test)

    # Convert the test embeddings and labels to tensors
    test_emb_tensor = torch.tensor(test_emb, dtype=torch.float32)
    test_lab_tensor = torch.tensor(test_lab, dtype=torch.long)

    # Create a TensorDataset for the test data and a DataLoader for batching
    test_ds = TensorDataset(test_emb_tensor, test_lab_tensor)
    test_dl = DataLoader(test_ds, config.training.batch_size, shuffle=False)
    
    # Instantiate the CNN and FF models
    cnn_model = CNNAudioClassifier(config.model.input_size,
                                   config.model.num_classes,
                                   config.model.dropout).to(device)
    ff_model = FFAudioClassifier(config.model.input_size,
                                 config.model.hidden_layers,
                                 config.model.num_classes,
                                 config.model.dropout).to(device)

    # Load the pre-trained model weights for the CNN, FF and SVM models
    cnn_model.load_state_dict(torch.load(f'{config.training.checkpoint_dir}/{config.best.cnn}', weights_only=True))
    print('CNN Model loaded.')
    ff_model.load_state_dict(torch.load(f'{config.training.checkpoint_dir}/{config.best.ff}', weights_only=True))
    print('FF Model loaded.')
    svm_model = joblib.load(f'{config.training.checkpoint_dir}/{config.best.svm}')
    print('SVM Model loaded.')

    criterion = nn.CrossEntropyLoss()                                   # Define the loss function for evaluation
    models = [('CNN', cnn_model), ('FF', ff_model)]                     # List of models to evaluate
    
    # Evaluate each model on the test dataset
    for model_name, model in models:
        test_metrics = evaluate(model, test_dl, criterion, device)      # Evaluate the model on the test dataset

        # Print out the evaluation metrics for each model
        for key, value in test_metrics.items():
            print(f'{model_name} Model Test {key}: {value:.4f}')

    # After evaluate CNN and FF models, use the FF model to extract features on test set for SVM
    ff_model.eval()
    with torch.no_grad():
        test_features = ff_model(test_emb_tensor.to(device)).cpu().numpy()

    # Evaluate SVM on test set
    test_predictions = svm_model.predict(test_features)
    svm_test_metrics = compute_metrics(test_predictions, test_lab)
    decision_scores = svm_model.decision_function(test_features)
    hinge_loss_value = hinge_loss(test_lab, decision_scores)
    for key, value in svm_test_metrics.items():
        print(f'\nSVM Model Test {key}: {value:.4f}', end='')
    print(f'\nSVM Model Test loss: {hinge_loss_value:.4f}')
