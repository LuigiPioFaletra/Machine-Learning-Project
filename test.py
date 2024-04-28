# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local application/library specific imports
from data_classes.mnist_dataset import MNISTDataset
from model_classes.ff_model import FFNN

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

def compute_metrics(predictions, references):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in dataloader:
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

if __name__ == '__main__':
    # Load configuration
    config = Dict(add_arguments())

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.training.device == 'cuda' else 'cpu')

    # Load data
    test_dataset = MNISTDataset(train=False, root=config.data.data_dir)
    test_dl = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    # Load model
    model = FFNN(
        input_size=28*28,
        hidden_layers=config.model.hidden_layers,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout
    )
    model.to(device)

    # Load model weights
    model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/best_model.pt"))
    print("Model loaded.")

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    test_metrics = evaluate(model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"Test {key}: {value:.4f}")
