import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# # Load audio from a file path using soundfile
# file_path = r"C:\Users\PC-12\Documents\vpc\2.wav"  # Replace with the actual path
# try:
#     audio_array, sample_rate = sf.read(file_path)

#     # Whisper expects a dictionary with 'array' and 'sampling_rate'
#     audio_data = {"array": audio_array, "sampling_rate": sample_rate}

#     result = pipe(audio_data)
#     print(result["text"])

# except Exception as e:
#     print(f"Error loading or processing audio: {e}")


# Example of handling stereo audio (if your file might have it)
file_path_stereo = r"C:\Users\PC-12\Documents\vpc\2.wav"
try:
    audio_array_stereo, sample_rate_stereo = sf.read(file_path_stereo)

    # Whisper expects a 1D array.  If stereo, convert to mono (simple average)
    if audio_array_stereo.ndim > 1:  # Check for stereo
        audio_array_mono = audio_array_stereo.mean(axis=1) # Average channels
    else:
        audio_array_mono = audio_array_stereo

    audio_data_stereo = {"array": audio_array_mono, "sampling_rate": sample_rate_stereo}
    result_stereo = pipe(audio_data_stereo)
    #print(result_stereo["text"])

except Exception as e:
    print(f"Error loading or processing stereo audio: {e}")

print(result_stereo["text"])