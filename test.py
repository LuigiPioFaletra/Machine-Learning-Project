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

from utils import compute_metrics, evaluate

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
