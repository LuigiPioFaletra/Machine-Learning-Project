import librosa
import numpy as np
import os
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def compute_metrics(predictions, references):
    acc = accuracy_score(references, predictions)                               # Compute accuracy of the predictions
    precision = precision_score(references, predictions, average='macro')       # Compute precision with macro averaging
    recall = recall_score(references, predictions, average='macro')             # Compute recall with macro averaging
    f1 = f1_score(references, predictions, average='macro')                     # Compute F1 score with macro averaging
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, dataloader, loss_fn, device):
    model.eval()                                                                # Set the model to evaluation mode
    running_loss = 0.0
    predictions = []
    references = []

    with torch.no_grad():                                                       # Disable gradient calculation for evaluation
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)                                   # Move data to the specified device
            outputs = model(x)                                                  # Get model predictions
            loss = loss_fn(outputs, y)                                          # Compute loss

            running_loss += loss.item()                                         # Accumulate loss
            preds = torch.argmax(outputs, dim=1)                                # Get the predicted class labels
            predictions.extend(preds.cpu().numpy())                             # Store predictions
            references.extend(y.cpu().numpy())                                  # Store true labels

    # Compute accuracy only if there are references
    if len(references) > 0:
        accuracy = (np.array(references) == np.array(predictions)).mean()
    else:
        accuracy = float('nan')                                                 # Set accuracy to NaN if there are no references

    return {
        "loss": running_loss / len(dataloader),                                 # Average loss over the dataset
        "accuracy": accuracy                                                    # Accuracy of the model
    }

def get_audio_path(root, track_id):
    folder = f'{int(track_id) // 1000:03d}'                                     # Folder based on track ID
    return os.path.join(root, folder, f'{int(track_id):06d}.mp3')               # Path to audio file

def get_genre_idx(metadata, genre_to_idx, idx):
    genre = metadata.iloc[idx]['track.7']                                       # Get the genre for the current track
    return genre_to_idx.get(genre, 0)                                           # Map the genre to an index

def get_track_id(metadata, idx):
    return metadata.iloc[idx]['Unnamed: 0']                                     # Get the track ID for the given index

def load_audio(audio_path, sample_rate):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)         # Use librosa to load the audio file
    except Exception as e:
        print(f"\nError loading file {audio_path}: {e}")                        # Move to the next sample if there's an error
        return None
    
    return torch.tensor(audio)                                                  # Convert to mono by averaging across the channels

def process_audio(audio, max_length):
    if audio.size(0) > max_length:
        audio = audio[:max_length]                                              # Truncate audio to the max length
    else:
        padding = max_length - audio.size(0)
        audio = torch.nn.functional.pad(audio, (0, padding))                    # Pad the audio if it's shorter

    return audio
