# Custom_Ai_Voice_Cloner_-_Text_To_Speech_Generator
!pip install TTS

import os
speaker_path = "LibriSpeech/dev-clean/84/121123"
files = os.listdir(speaker_path)
print(f"🎤 Total files for speaker 84-121123: {len(files)}")
print("🔊 Example files:", files[:5])


from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

input_dir = "LibriSpeech/dev-clean/84/121123"
output_dir = "clean_wavs"
os.makedirs(output_dir, exist_ok=True)

def preprocess_audio(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".flac"):
            filepath = os.path.join(input_folder, filename)
            sound = AudioSegment.from_file(filepath, format="flac")

            # Trim silence
            chunks = split_on_silence(sound, silence_thresh=-40, min_silence_len=300, keep_silence=100)
            # Save each chunk as .wav
            for i, chunk in enumerate(chunks):
                out_path = os.path.join(output_folder, f"{filename[:-5]}_chunk{i}.wav")
                chunk = chunk.set_frame_rate(22050).set_channels(1)  # mono, 22050 Hz
                chunk.export(out_path, format="wav")
preprocess_audio(input_dir, output_dir)

# 🔽 Download pre-trained TTS & vocoder models from Coqui
!tts --text "Assalamualaikum! Yeh aik test hai AI voice cloning ka." \
    --model_name "tts_models/en/ljspeech/tacotron2-DDC" \
    --vocoder_name "vocoder_models/en/ljspeech/waveglow" \
    --out_path "output.wav"


!pip install gradio
import gradio as gr
from TTS.api import TTS
# Load pre-trained model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
# Gradio function
def generate_audio(text):
    tts.tts_to_file(text=text, file_path="voice.wav")
    return "voice.wav"
# Create Web UI
gr.Interface(fn=generate_audio, inputs="text", outputs="audio", title="🎤 AI Voice Cloner").launch()
