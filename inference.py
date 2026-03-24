import base64
import io
import numpy as np

import torch.cuda
import requests
from model import AudioCNN
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AudioProcessor:

    def __init__(self):
        self.transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)

        return spectrogram.unsqueeze(0)


