from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        data, sample_rate = sf.read(audio_path)

        waveform = torch.tensor(data, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
        else:
            waveform = waveform.T  # [channels, samples]

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim = 0, keepdim = True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']