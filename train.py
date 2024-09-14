import joblib
import numpy as np
import os
import torch
import torch.nn as nn

from addict import Dict
from data_classes.fma_dataset import FMADataset
from extract_representations.audio_embeddings import AudioEmbeddings
from model_classes.cnn_model import CNNAudioClassifier
from model_classes.ffnn_model import FFNNAudioClassifier
from sklearn.metrics import hinge_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import compute_metrics, evaluate, extract_and_preprocess_data, save_data
from yaml_config_override import add_arguments

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()                                               # Set the model to training mode                      
    running_loss = 0.0
    predictions = []
    references = []
    
    # Loop over batches in the dataloader
    for i, (x, y) in enumerate(tqdm(dataloader, desc='\nTraining')):
        x, y = x.to(device), y.to(device)                       # Move data to the specified device
        optimizer.zero_grad()                                   # Zero the gradients
        outputs = model(x)                                      # Forward pass: compute model predictions
        loss = loss_fn(outputs, y)                              # Compute the loss
        loss.backward()                                         # Backward pass: compute gradients
        optimizer.step()                                        # Update model weights
        scheduler.step()                                        # Update learning rate if scheduler is defined
        running_loss += loss.item()                             # Accumulate the loss
        pred = torch.argmax(outputs, dim=1)                     # Get predicted class labels
        predictions.extend(pred.cpu().numpy())                  # Store predictions
        references.extend(y.cpu().numpy())                      # Store true labels

    train_metrics = compute_metrics(predictions, references)    # Compute training metrics
    train_metrics['loss'] = running_loss / len(dataloader)      # Average loss
    return train_metrics                                        # Return computed metrics
    
def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric
        
    # If the current model is the best, update the best model and metric
    if is_best:
        print(f'New best model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}')
        best_val_metric = val_metrics[evaluation_metric]
        best_model = model
        
    return best_val_metric, best_model

def manage_early_stopping(val_metric, best_val_metric, early_stopping_counter, patience):
    # Check if the validation metric has improved
    if val_metric <= best_val_metric:
        early_stopping_counter = 0      # Reset counter if improvement is seen
    else:
        early_stopping_counter += 1     # Increment counter if no improvement

    # If the counter exceeds patience, trigger early stopping
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        return True

    return False

def load_or_extract_data(root, metadata_file, sample_rate, max_duration, batch_size, device, directory, split):
    # Define file paths for embeddings and labels
    embedding_file = f'{directory}/{split}_embeddings.npy'
    label_file = f'{directory}/{split}_labels.npy'
    
    # Check if the embeddings and labels already exist as .npy files
    if os.path.exists(embedding_file) and os.path.exists(label_file):
        # Load pre-saved embeddings and labels
        embeddings = np.load(embedding_file)
        labels = np.load(label_file)
    else:
        model = AudioEmbeddings()       # Initialize the audio embedding extraction model
        
        # Create the dataset and dataloader for the specified split
        dataset = FMADataset(root, metadata_file, split, sample_rate, max_duration)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'training'))
        
        # Extract embeddings and labels from the audio data
        embeddings, labels = extract_and_preprocess_data(model, dataloader, device, split)
        
        # Save the extracted embeddings and labels as .npy files for future use
        save_data(embeddings, labels, split, directory)
    
    return embeddings, labels           # Return the embeddings and labels


if __name__ == '__main__':
    config = Dict(add_arguments())      # Load configuration from YAML file
    device = torch.device('cuda' if torch.cuda.is_available() and config.training.device == 'cuda' else 'cpu')

    # Check if training embeddings and labels already exist and upload them, otherwise extract and save them
    train_emb, train_lab = load_or_extract_data(config.data.dataset_dir,
                                                config.data.metadata_file,
                                                config.training.sample_rate,
                                                config.training.max_duration,
                                                config.training.batch_size,
                                                config.training.device,
                                                config.training.npy_dir,
                                                config.split.train)

    # Check if validation embeddings and labels already exist and upload them, otherwise extract and save them
    val_emb, val_lab = load_or_extract_data(config.data.dataset_dir,
                                            config.data.metadata_file,
                                            config.training.sample_rate,
                                            config.training.max_duration,
                                            config.training.batch_size,
                                            config.training.device,
                                            config.training.npy_dir,
                                            config.split.val)
    
    # Convert embeddings and labels to tensors
    train_emb_tensor = torch.tensor(train_emb, dtype=torch.float32)
    train_lab_tensor = torch.tensor(train_lab, dtype=torch.long)
    val_emb_tensor = torch.tensor(val_emb, dtype=torch.float32)
    val_lab_tensor = torch.tensor(val_lab, dtype=torch.long)
    
    # Create TensorDatasets for training and validation sets
    train_ds = TensorDataset(train_emb_tensor, train_lab_tensor)
    val_ds = TensorDataset(val_emb_tensor, val_lab_tensor)
    
    # Create DataLoaders for the datasets
    train_dl = DataLoader(train_ds, config.training.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, config.training.batch_size, shuffle=False)
    
    # Instantiate the CNN and FFNN models
    cnn_model = CNNAudioClassifier(config.model.input_size,
                                   config.model.num_classes,
                                   config.model.dropout).to(device)
    ffnn_model = FFNNAudioClassifier(config.model.input_size,
                                     config.model.hidden_layers,
                                     config.model.num_classes,
                                     config.model.dropout).to(device)
    
    cnn_model.to(device)
    ffnn_model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    ffnn_optimizer = torch.optim.Adam(ffnn_model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    # Learning rate scheduler for warmup and decay
    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    # Warmup + linear decay schedule
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    cnn_scheduler = torch.optim.lr_scheduler.LambdaLR(cnn_optimizer, lr_lambda=scheduler_lambda)
    ffnn_scheduler = torch.optim.lr_scheduler.LambdaLR(ffnn_optimizer, lr_lambda=scheduler_lambda)

    # List of models to train
    models = [('CNN', cnn_model, cnn_optimizer, cnn_scheduler, config.best.cnn),
              ('FFNN', ffnn_model, ffnn_optimizer, ffnn_scheduler, config.best.ff)]

    # Inizialization of early stopping parameters
    patience = 3
    early_stopping_counter = 0

    for model_name, model, optimizer, scheduler, best_model in models:
        print(f'\n\nStart {model_name} model training and validation')

        # Initialize best validation metric and model
        best_val_metric = float('inf') if config.training.best_metric_lower_is_better else float('-inf')
        best_model = None
        
        for epoch in range(config.training.epochs):
            print(f'\nEpoch {epoch+1}/{config.training.epochs}')
            
            # Train for one epoch and evaluate on validation set
            train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
            val_metrics = evaluate(model, val_dl, criterion, device)
            print(f'Train loss: {train_metrics["loss"]:.4f} - Train accuracy: {train_metrics["accuracy"]:.4f}')
            print(f'Val loss: {val_metrics["loss"]:.4f} - Val accuracy: {val_metrics["accuracy"]:.4f}')
            
            # Update best model if validation performance improves
            best_val_metric, best_model = manage_best_model_and_metrics(
                model, 
                config.training.evaluation_metric, 
                val_metrics, 
                best_val_metric, 
                best_model, 
                config.training.best_metric_lower_is_better
            )

            # Check early stopping conditions
            if manage_early_stopping(val_metrics['loss'], best_val_metric, early_stopping_counter, patience):
                break                   # Stop training if early stopping is triggered

        # Save the best model
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        torch.save(best_model.state_dict(), f'{config.training.checkpoint_dir}/best_{model_name.lower()}_model.pt')
        print('Model saved.')

    print(f'\n\nStart SVM model grid search and validation\n')
    svm_model = SVC()                   # Instantiate the SVM model
    
    # After training CNN and FFNN models, use the FFNN model to extract features on training and validation sets for SVM
    ffnn_model.eval()
    with torch.no_grad():
        train_features = ffnn_model(torch.tensor(train_emb, dtype=torch.float32).to(device)).cpu().numpy()
        val_features = ffnn_model(torch.tensor(val_emb, dtype=torch.float32).to(device)).cpu().numpy()
    
    # Parameter distribution for SVM randomized search
    param_dist = {
        'C': [0.1, 1, 10, 100],         # Regularization parameter
        'gamma': ['scale', 'auto'],     # Kernel coefficient for 'rbf'
        'kernel': ['linear', 'rbf']     # Kernel type
    }

    # Randomized search across the specified hyperparameter distributions
    random_search = RandomizedSearchCV(svm_model, param_dist, n_iter=10, cv=5, scoring='accuracy', verbose=3)
    for _ in tqdm(range(1)):
        random_search.fit(train_features, train_lab)
    best_svm = random_search.best_estimator_
    best_parameters = random_search.best_params_
    print(f'\nBest SVM: {best_parameters}')

    # Save the best SVM model
    joblib.dump(best_svm, f'{config.training.checkpoint_dir}/{config.best.svm}')
    print('Model saved.')

    # Evaluate SVM on validation set
    val_predictions = best_svm.predict(val_features)
    svm_val_metrics = compute_metrics(val_predictions, val_lab)
    decision_scores = best_svm.decision_function(val_features)
    hinge_loss_value = hinge_loss(val_lab, decision_scores)
    print(f'\nVal loss: {hinge_loss_value:.4f} - Val accuracy: {svm_val_metrics["accuracy"]:.4f}')
