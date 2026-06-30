import torch
import whisper
import librosa
import soundfile as sf

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from transformers import pipeline

from llama_cpp import Llama

# CHECK GPU

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# FILE PICKER
# =========================

print("\nSelect Audio File...")

Tk().withdraw()

audio_path = askopenfilename(
    title="Select Audio File",
    filetypes=[
        ("Audio Files", "*.mp3 *.wav *.mpeg *.m4a")
    ]
)

print("\nSelected File:")
print(audio_path)


# LOAD AUDIO

print("\nLoading Audio...")

audio, sr = librosa.load(audio_path, sr=16000)

print("Audio Loaded Successfully")
print("Sample Rate:", sr)
print("Audio Length:", len(audio)/sr, "seconds")


# SAVE PROCESSED AUDIO

processed_audio = "processed_audio.wav"

sf.write(processed_audio, audio, sr)

print("Processed audio saved")


# WHISPER TRANSCRIPTION

print("\nLoading Whisper Tiny Model...")

model = whisper.load_model("tiny")

if torch.cuda.is_available():
    model = model.to("cuda")

print("Whisper Model Loaded")

print("\nTranscribing Audio...\n")

result = model.transcribe(processed_audio)

transcription = result["text"]


# FULL TRANSCRIPTION

print("\n===== FULL AUDIO TRANSCRIPTION =====\n")
print(transcription)


# EMOTION DETECTION

print("\nLoading Emotion Detection Model...")

emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=0 if torch.cuda.is_available() else -1
)

emotion = emotion_classifier(processed_audio)

print("\n===== EMOTION ANALYSIS =====")
print(emotion[:3])


# AUDIO SCENE DETECTION

print("\nLoading Audio Scene Model...")

audio_classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device=-1
)

scene = audio_classifier(processed_audio)

print("\n===== AUDIO SCENE =====")
print(scene[:5])


# LOAD TINYLLAMA

print("\nLoading TinyLlama Reasoning Model...\n")

llm = Llama(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8
)

print("TinyLlama Loaded Successfully")


# PREPARE CHAT PROMPT

scene_labels = [s['label'] for s in scene[:5]]

messages = [
    {
        "role": "system",
        "content": (
            "You are an advanced Audio Language Model. "
            "Analyze audio transcription, emotion detection, and audio scene labels carefully. "
            "Always answer in this exact format:\n\n"
            "1. Number of speakers:\n"
            "2. Conversation summary:\n"
            "3. Audio scene:\n"
            "4. Final reasoning:\n"
            "5. Emotion analysis:\n\n"
            "Give intelligent reasoning directly from the transcription."
        )
    },
    {
        "role": "user",
        "content": f"""
TRANSCRIPTION:
{transcription}

EMOTION ANALYSIS:
{emotion}

AUDIO SCENE:
{scene_labels}
"""
    }
]

# GENERATE FINAL REASONING

print("\nGenerating Final Reasoning...\n")

output = llm.create_chat_completion(
    messages=messages,
    max_tokens=400,
    temperature=0.5
)

final_reasoning = output["choices"][0]["message"]["content"]

# FINAL OUTPUT

print("\n===== FINAL REASONING =====\n")

print(final_reasoning)

# CLOSE MODEL

llm.close()