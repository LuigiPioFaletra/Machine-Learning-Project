import numpy as np
import os
import pandas as pd
import sys
import torch

from torch.utils.data import Dataset

# Append the parent directory to the system path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_audio_path, get_genre_idx, get_track_id, load_audio, process_audio

class FMADataset(Dataset):
    def __init__(self, root, metadata_file, data_split, sample_rate, max_duration):
        self.root = root                                                            # Path to the root directory of audio files
        df = pd.read_csv(metadata_file, low_memory=False)                           # Load metadata from the CSV file
        self.metadata = df[(df['set'] == data_split) & (df['set.1'] == 'small')]    # Filter metadata based on the data split and subset ('small')
        self.sample_rate = sample_rate                                              # Desired sample rate for audio files
        self.max_length = sample_rate * max_duration                                # Maximum length of audio in samples, based on the duration
        
        # Get unique genres and create a mapping from genre name to an index
        self.genres = np.sort(self.metadata['track.7'].unique())
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the track ID, audio file path, and genre index for the given sample index
        track_id = get_track_id(self.metadata, idx)
        audio_path = get_audio_path(self.root, track_id)
        genre_idx = get_genre_idx(self.metadata, self.genre_to_idx, idx)

        # Load the audio file and handle any errors that occur during loading
        audio = load_audio(audio_path, self.sample_rate)    # During loading, converts the audio to mono and resamples to the sample_rate value
        if audio is None:                                   # If there was an error loading the audio, try the next sample
            return self.__getitem__((idx + 1) % len(self.metadata))

        audio = process_audio(audio, self.max_length)       # Process the audio: truncate or pad to the desired length
        return audio, genre_idx                             # Return the processed audio and its corresponding genre index
