# Standard library imports
import os

# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local application/library specific imports
from yaml_config_override import add_arguments
from addict import Dict
from data_classes.mnist_dataset import MNISTDataset
from model_classes.ff_model import FFNN

def compute_metrics(predictions, references):
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='macro')
    recall = recall_score(references, predictions, average='macro')
    f1 = f1_score(references, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    predictions = []
    references = []
    
    for i, batch in enumerate(tqdm(dataloader, desc='Training')):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())
        
    train_metrics = compute_metrics(predictions, references)
    train_metrics['loss'] = running_loss / len(dataloader)
    
    return train_metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())
            
    val_metrics = compute_metrics(predictions, references)
    val_metrics['loss'] = running_loss / len(dataloader)
    
    return val_metrics

def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric
        
    if is_best:
        print(f"New best model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        best_val_metric = val_metrics[evaluation_metric]
        best_model = model
        
    return best_val_metric, best_model

if __name__ == '__main__':
    
    # ---------------------
    # 1. Parse configuration
    # ---------------------
    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Load data
    # ---------------------
    
    train_dataset = MNISTDataset(train=True, root=config.data.data_dir)
    test_dataset  = MNISTDataset(train=False, root=config.data.data_dir)
    
    train_size = int(config.data.train_ratio * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, len(train_dataset) - train_size]
    )
    
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # print some statistics
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # ---------------------
    # 3. Load model
    # ---------------------
    
    model = FFNN(
        input_size=28*28,
        hidden_layers=config.model.hidden_layers,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout
    )
    
    if config.training.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("CUDA not available. Using CPU.")
        device = torch.device('cpu')
        
    model.to(device)
    
    # ---------------------
    # 4. Train model
    # ---------------------
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # learning rate scheduler
    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    # warmup + linear decay
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    
    if config.training.best_metric_lower_is_better:
        best_val_metric = float('inf')
    else:
        best_val_metric = float('-inf')
        
    best_model = None
    
    for epoch in range(config.training.epochs):
        print(f"Epoch {epoch+1}/{config.training.epochs}")
        
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_dl, criterion, device)
        
        print(f"Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")
        
        best_val_metric, best_model = manage_best_model_and_metrics(
            model, 
            config.training.evaluation_metric, 
            val_metrics, 
            best_val_metric, 
            best_model, 
            config.training.best_metric_lower_is_better
        )
        
    # --------------------------------
    # 5. Evaluate model on test set
    # --------------------------------
    
    test_metrics = evaluate(best_model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")
        
    # ---------------------
    # 6. Save model
    # ---------------------
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/best_model.pt")
    
    print("Model saved.")
    
    
