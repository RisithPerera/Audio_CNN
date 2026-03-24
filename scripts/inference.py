import base64
import io

import librosa
import numpy as np
import soundfile as sf
import torch.cuda
import torch.nn as nn
import torchaudio.transforms as T
from fastapi import FastAPI
from pydantic import BaseModel

from model import AudioCNN

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


class InferenceRequest(BaseModel):
    audio_data: str


class AudioClassifier:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('../models/audio_cnn.pth', map_location=self.device)
        self.classes = checkpoint['classes']
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print('Model Loaded')

    def inference(self, audio_data, sample_rate):
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 22050:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)

        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output = self.model(spectrogram)
            output = torch.nan_to_num(output)

            probabilities = torch.softmax(output, dim = 1)

            top3_probs, top3_indicies = torch.topk(probabilities[0], 3)

            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indicies)]

            return predictions

classifier = None

@app.on_event("startup")
def load():
    global classifier
    classifier = AudioClassifier()


@app.post("/inference/")
def process_inference_request(request: InferenceRequest):
    audio_byte = base64.b64decode(request.audio_data)

    audio_data, sample_rate = sf.read(io.BytesIO(audio_byte), dtype="float32")

    predictions = classifier.inference(audio_data, sample_rate)

    response = {
        "predictions": predictions
    }

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)