import pyaudio
from transformers.models.wav2vec2 import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.models.hubert import HubertForCTC
import torch
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
import threading
import time
from collections import deque
from .distilbert import DistilBertWithWeights

class RealTimeSentimentAnalyzerW2V:
    def __init__(self, input_device_index=1, rate=16000, chunk=768, seconds=6, overlap=1.5, context_window=6, output_queue=None):
        # Audio parameters
        self.INPUT_DEVICE_INDEX = input_device_index
        self.RATE = rate
        self.CHUNK = chunk
        self.SECONDS = seconds
        self.OVERLAP_SCND = overlap
        self.CONTEXT_WINDOW = context_window
        self.BUFFER_SIZE = int(rate * seconds)
        self.STEP_SIZE = int(rate * (seconds - overlap))

        # Shared data structures
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        self.context_window = deque(maxlen=self.CONTEXT_WINDOW)
        self.context_lock = threading.Lock()

        # Runtime state
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        self.output_queue = output_queue

        # Load models and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert_final_modelV2")
        self.sentiment_model = DistilBertWithWeights(num_labels=3).to(self.device).eval()

        state_dict = torch.load("distilbert_final_modelV2/pytorch_model.bin", map_location=self.device)
        state_dict.pop("loss_fct.weight", None)
        self.sentiment_model.load_state_dict(state_dict)

    def start(self):
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self._audio_capture, daemon=True)
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.capture_thread.start()
            self.processing_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.capture_thread.join()
            self.processing_thread.join()

    def _audio_capture(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.RATE,
                        input=True,
                        input_device_index=self.INPUT_DEVICE_INDEX,
                        frames_per_buffer=self.CHUNK)
        print("🎙️ Audio capture started...")
        try:
            while self.running:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                with self.buffer_lock:
                    self.audio_buffer.extend(samples)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _process_audio(self):
        while self.running:
            time.sleep(self.SECONDS - self.OVERLAP_SCND)
            with self.buffer_lock:
                if len(self.audio_buffer) < self.BUFFER_SIZE:
                    continue

                samples = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
                reduced = list(self.audio_buffer)[self.STEP_SIZE:]
                self.audio_buffer.clear()
                self.audio_buffer.extend(reduced)

            # Transcription
            inputs = self.processor(samples, sampling_rate=self.RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.asr_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].lower()

            with self.context_lock:
                self.context_window.append(transcription)
                context_text = " ".join(self.context_window)

            # Sentiment
            inputs = self.tokenizer(context_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs["logits"]
                predicted_class = torch.argmax(logits, dim=-1).item()
            label_map = {0: "❌ negative", 1: "😐 neutral", 2: "✅ positive"}

            output_line = f"🗣 {transcription}\n→ Sentiment: {label_map[predicted_class]}"
            print(output_line)

            if self.output_queue:
                self.output_queue.put((transcription, label_map[predicted_class]))



class RealTimeSentimentAnalyzerHBRT:
    def __init__(self, input_device_index=1, rate=16000, chunk=768, seconds=6, overlap=1.5, context_window=6, output_queue=None):
        # Audio parameters
        self.INPUT_DEVICE_INDEX = input_device_index
        self.RATE = rate
        self.CHUNK = chunk
        self.SECONDS = seconds
        self.OVERLAP_SCND = overlap
        self.CONTEXT_WINDOW = context_window
        self.BUFFER_SIZE = int(rate * seconds)
        self.STEP_SIZE = int(rate * (seconds - overlap))

        # Shared data structures
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        self.context_window = deque(maxlen=self.CONTEXT_WINDOW)
        self.context_lock = threading.Lock()

        # Runtime state
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        self.output_queue = output_queue

        # Load models and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert_final_modelV2")
        self.sentiment_model = DistilBertWithWeights(num_labels=3).to(self.device).eval()

        state_dict = torch.load("distilbert_final_modelV2/pytorch_model.bin", map_location=self.device)
        state_dict.pop("loss_fct.weight", None)
        self.sentiment_model.load_state_dict(state_dict)

    def start(self):
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self._audio_capture, daemon=True)
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.capture_thread.start()
            self.processing_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.capture_thread.join()
            self.processing_thread.join()

    def _audio_capture(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.RATE,
                        input=True,
                        input_device_index=self.INPUT_DEVICE_INDEX,
                        frames_per_buffer=self.CHUNK)
        print("🎙️ Audio capture started...")
        try:
            while self.running:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                with self.buffer_lock:
                    self.audio_buffer.extend(samples)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _process_audio(self):
        while self.running:
            time.sleep(self.SECONDS - self.OVERLAP_SCND)
            with self.buffer_lock:
                if len(self.audio_buffer) < self.BUFFER_SIZE:
                    continue

                samples = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
                reduced = list(self.audio_buffer)[self.STEP_SIZE:]
                self.audio_buffer.clear()
                self.audio_buffer.extend(reduced)

            # Transcription
            inputs = self.processor(samples, sampling_rate=self.RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.asr_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].lower()

            with self.context_lock:
                self.context_window.append(transcription)
                context_text = " ".join(self.context_window)

            # Sentiment
            inputs = self.tokenizer(context_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs["logits"]
                predicted_class = torch.argmax(logits, dim=-1).item()
            label_map = {0: "❌ negative", 1: "😐 neutral", 2: "✅ positive"}

            output_line = f"🗣 {transcription}\n→ Sentiment: {label_map[predicted_class]}"
            print(output_line)

            if self.output_queue:
                self.output_queue.put((transcription, label_map[predicted_class]))