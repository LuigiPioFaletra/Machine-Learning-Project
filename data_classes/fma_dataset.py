from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import pandas as pd
import torch

class FMADataset(Dataset):
    def __init__(self, root, metadata_file, data_split, sample_rate=16000, max_duration=30):
        self.root = root
        df = pd.read_csv(metadata_file, low_memory=False)   # Load metadata from CSV file
        self.metadata = df[(df['set'] == data_split) & (df['set.1'] == 'small')]        # Filter rows
        self.sample_rate = sample_rate
        self.max_length = sample_rate * max_duration        # Maximum length in samples
        self.genres = np.sort(self.metadata['track.7'].unique())        # Create a sorted list of genres
        self.genre_to_idx = {genre: idx + 1 for idx, genre in enumerate(self.genres)}   # Map from genre names to indices, starting from 1

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        track_id = self.metadata.iloc[idx]['Unnamed: 0']    # 'Unnamed: 0' contains the track_id
        folder = f'{int(track_id) // 1000:03d}'             # Determine the directory by dividing track_id by 1000
        audio_path = os.path.join(self.root, folder, f'{int(track_id):06d}.mp3')        # Construct the full path to the audio file

        # Handling execution of missing files
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)         # Load audio, convert to mono and resample to sample_rate
        except (FileNotFoundError, IOError) as e:
            print(f"File {audio_path} not found or cannot be read. Skipping this entry.")
            return self.__getitem__((idx + 1) % len(self.metadata))     # Handle missing file by skipping it and trying the next entry

        # Preprocessing audio to a fixed length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        genre = self.metadata.iloc[idx]['track.7']          # 'track.7' contains the genre as a name
        genre_idx = self.genre_to_idx.get(genre, 0)         # Convert genre to numeric index (0 if not found)        
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)   # Adding channel size
        return audio, genre_idx
