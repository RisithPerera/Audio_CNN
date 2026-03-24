import base64
import io

import requests
import soundfile as sf


def send_request():
    print("Reading audio...")
    audio_data, sample_rate = sf.read("../dataset/ESC-50/audio/1-100038-A-14.wav")
    buffer = io.BytesIO()

    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {"audio_data": audio_b64}
    url = r"http://localhost:8000/inference/"

    print("Sending request...")
    response = requests.post(url, json = payload, timeout = 60)
    print("Response received!")

    response.raise_for_status()
    result = response.json()

    print("Top predictions: ")
    for pred in result.get("predictions", []):
        print(f" -{pred['class']} {pred['confidence']:0.2%}")


# Start the training
if __name__ == "__main__":
    send_request()