from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from dataset import ESC50Dataset
from model import AudioCNN


def evaluation():
    esc50_dir = Path("../dataset/ESC-50")

    val_transform = nn.Sequential(
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

    val_dataset = ESC50Dataset(
        data_dir = esc50_dir,
        metadata_file = esc50_dir/"meta"/"esc50.csv",
        split='val',
        transform=val_transform
    )

    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('../models/audio_cnn.pth', map_location=device)
    model = AudioCNN(num_classes=len(val_dataset.classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    evaluation()