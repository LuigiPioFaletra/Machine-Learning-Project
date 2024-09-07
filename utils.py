import librosa
import numpy as np
import os
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
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

def evaluate(model, dataloader, criterion, device):
    model.eval()                                                                # Set the model to evaluation mode                
    running_loss = 0.0                                                          # Initialize running loss
    predictions = []                                                            # List to store model predictions
    references = []                                                             # List to store true labels

    with torch.no_grad():                                                       # Disable gradient calculation for evaluation
        for i, (x, y) in enumerate(tqdm(dataloader, desc='\nValidation')):      # Progress bar for validation
            x, y = x.to(device), y.to(device)                                   # Move data to the specified device (CPU or GPU)
            outputs = model(x)                                                  # Get model predictions
            loss = criterion(outputs, y)                                        # Compute loss using the provided criterion
            running_loss += loss.item()                                         # Accumulate the loss
            pred = torch.argmax(outputs, dim=1)                                 # Get the predicted class labels (with highest score)
            predictions.extend(pred.cpu().numpy())                              # Convert predictions to numpy and store
            references.extend(y.cpu().numpy())                                  # Convert true labels to numpy and store
            
    val_metrics = compute_metrics(predictions, references)                      # Calculate evaluation metrics
    val_metrics['loss'] = running_loss / len(dataloader)                        # Compute average loss
    return val_metrics

def get_audio_path(root, track_id):
    folder = f'{int(track_id) // 1000:03d}'                                     # Folder structure is based on track ID
    return os.path.join(root, folder, f'{int(track_id):06d}.mp3')               # Return full path to the audio file

def get_genre_idx(metadata, genre_to_idx, idx):
    genre = metadata.iloc[idx]['track.7']                                       # Get the genre from metadata (assuming 'track.7' is the genre column)
    return genre_to_idx.get(genre, 0)                                           # Map the genre to an index using genre_to_idx dictionary

def get_track_id(metadata, idx):
    return metadata.iloc[idx]['Unnamed: 0']                                     # Get the track ID from metadata (assuming 'Unnamed: 0' is the track ID column)

def load_audio(audio_path, sample_rate):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)         # Load the audio file in mono format
    except Exception as e:
        print(f'\nError loading file {audio_path}: {e}')                        # Print an error message if the file fails to load
        return None                                                             # Return None if there's an error
    
    return torch.tensor(audio)                                                  # Convert the loaded audio to a PyTorch tensor

def process_audio(audio, max_length):
    if audio.size(0) > max_length:
        audio = audio[:max_length]                                              # Truncate the audio to the maximum length
    else:
        padding = max_length - audio.size(0)
        audio = torch.nn.functional.pad(audio, (0, padding))                    # Pad the audio with zeros if it's shorter than max_length

    return audio
        
def extract_and_preprocess_data(model, dataloader, device, split):
    model.model.eval()                                                          # Set the model to evaluation mode
    embeddings = []                                                             # List to store extracted embeddings
    labels = []                                                                 # List to store corresponding labels

    with torch.no_grad():                                                       # Disable gradient calculation
        for x, y in tqdm(dataloader, desc=f'\n{split.capitalize()} embeddings and labels extracting'):
            x = x.numpy()                                                       # Convert input data to numpy array
            x = [model.extract(speech=audio_file) for audio_file in x]          # Extract embeddings for each audio file using the model
            x = np.array(x)                                                     # Convert list of embeddings to numpy array
            x = np.mean(x, axis=1)                                              # Average the embeddings along the time axis
            embeddings.append(x)                                                # Store the embeddings
            labels.append(y.numpy())                                            # Store the labels

    embeddings = np.concatenate(embeddings, axis=0)                             # Concatenate all embeddings
    labels = np.concatenate(labels, axis=0)                                     # Concatenate all labels
    
    # Normalize embeddings
    scaler = StandardScaler()                                                   # Initialize a standard scaler
    embeddings = scaler.fit_transform(embeddings)                               # Use fit_transform method to normalize the embeddings

    return embeddings, labels

def save_data(embeddings, labels, prefix, directory):
    os.makedirs(directory, exist_ok=True)
    np.save(f'{directory}/{prefix}_embeddings.npy', embeddings)                 # Save embeddings as a numpy file
    np.save(f'{directory}/{prefix}_labels.npy', labels)                         # Save labels as a numpy file
    print(f'{prefix.capitalize()} embeddings and labels saved in .npy files')
