from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from model import AudioCNN
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim = 0, keepdim = True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']


def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, prediction, y_a, y_b, lam):
    return lam * criterion(prediction, y_a) + (1 - lam) * criterion(prediction, y_b)


def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    esc50_dir = Path("dataset/ESC-50")

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30), # Dropout technique for audio
        T.TimeMasking(time_mask_param=80)
    )

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

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split='train',
        transform=train_transform
    )

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split='val',
        transform=val_transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Shuffle OFF because we can check test results with last time test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100

    # Label smoothing = [1, 0, 0, 0, 0] -> [0.9, 0.025, 0.025, 0.025, 0.025]
    # Otherwise makes the model overconfident — it pushes predictions toward extreme values,
    # which hurts generalization. This is called overfitting to labels.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # lr = Controls how big each update step is
    # weight_decay = Adds a small penalty for having large weights, pushing them toward zero
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader), # Number of batches in one epoch
        pct_start = 0.1,

    )

    best_accuracy = 0.0

    print("Starting training...!")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data,target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rates', optimizer.param_groups[0]['lr'], epoch)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(f'Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


# Start the training
if __name__ == "__main__":
    train()