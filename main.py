import sys
import pyaudio
from transformers.models.wav2vec2 import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
import threading
import time
from collections import deque
from models import DistilBertWithWeights

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
asr_model.eval()

# Load sentiment model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert_final_model")

# Load model architecture
sentiment_model = DistilBertWithWeights(num_labels=3)

# Load trained weights
state_dict = torch.load("distilbert_final_model/pytorch_model.bin", map_location=torch.device("cpu"))

# Remove non-model layer parameters
state_dict.pop("loss_fct.weight", None)

# Load weights into model
sentiment_model.load_state_dict(state_dict)

sentiment_model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)
sentiment_model.to(device)

# Audio parameters
INPUT_DEVICE_INDEX = 1  # Change this if needed
RATE = 16000
CHUNK = 512
SECONDS = 5
BUFFER_SIZE = int(RATE * SECONDS)  # 10 seconds of audio
OVERLAP_SCND = 1
STEP_SIZE = int(RATE * (SECONDS - OVERLAP_SCND)) # OVERLAP_SCND seconds overlap

# Shared buffer
audio_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()

#Context buffer
CONTEXT_WINDOW = 6
context_window = deque(maxlen=CONTEXT_WINDOW)
context_lock = threading.Lock()

running = True

def audio_capture():
    global running
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    print("🎙️ Audio capture started...")
    try:
        while running:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)

            with buffer_lock:
                audio_buffer.extend(samples)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_audio():
    global running
    while running:
        time.sleep(SECONDS-OVERLAP_SCND)
        with buffer_lock:
            if len(audio_buffer) < BUFFER_SIZE:
                continue  # Not enough data yet

            # normalizing 
            samples = np.array(audio_buffer, dtype=np.float32) / 32768.0
          
            reduced = list(audio_buffer)[STEP_SIZE:]
            audio_buffer.clear()
            audio_buffer.extend(reduced)


        # ASR
        inputs = processor(samples, sampling_rate=RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = asr_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower()

        print(f"\n🗣 Transcription: {transcription}")

        # Maintain context
        with context_lock:
            context_window.append(transcription)
            context_text = " ".join(context_window)

        # Sentiment Analysis with context
        inputs = tokenizer(context_text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs["logits"]
            predicted_class = torch.argmax(logits, dim=-1).item()

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        print(f"Sentiment Analysis: {label_map[predicted_class]}")



capture_thread = threading.Thread(target=audio_capture)
processing_thread = threading.Thread(target=process_audio)

capture_thread.start()
processing_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    running = False
    processing_thread.join()
    capture_thread.join()
    sys.exit(0) 