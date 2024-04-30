import torch
from PIL import Image
from io import BytesIO
import requests

import librosa

from audio_embeddings import AudioEmbeddings
from text_embeddings import TextEmbeddings
from vision_embeddings import VisionEmbeddings

def test_audio_embeddings():
    audio_url = "https://download.samplelib.com/mp3/sample-3s.mp3"
    
    # download the audio file
    audio_file_path = "sample-3s.mp3"
    response = requests.get(audio_url)
    with open(audio_file_path, "wb") as audio_file:
        audio_file.write(response.content)
        
    # load the audio file
    audio, sr = librosa.load(audio_file_path, sr=16000)

    extractor = AudioEmbeddings(model_name='ALM/hubert-base-audioset')
    audio_vector = extractor.extract(audio)
    print("Audio Embeddings:", audio_vector.shape, type(audio_vector))

def test_text_embeddings():
    text = "Hello, world! This is a test text for embeddings."
    text_embeddings = TextEmbeddings(model_name='bert-base-multilingual-cased')
    text_vector = text_embeddings.extract(text)
    print("Text Embeddings:", text_vector.shape, type(text_vector))

def test_vision_embeddings():
    # Example image URL
    image_url = "https://educationaround.org/wp-content/uploads/2021/01/img-unikore-FULL-1024x576.png"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")
    
    vision_embeddings = VisionEmbeddings(model_name='google/vit-base-patch16-224')
    image_vector = vision_embeddings.extract(image)
    print("Image Embeddings:", image_vector.shape, type(image_vector))
    
    
    
if __name__ == "__main__":
    # test_audio_embeddings()
    # test_text_embeddings()
    test_vision_embeddings()