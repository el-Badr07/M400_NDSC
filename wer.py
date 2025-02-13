from evaluate import load
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import torchaudio
import librosa

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

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]
#print(sample)
sample1=r'C:\Users\PC-12\Documents\vpc\speech.wav'
sample2=r'C:\Users\PC-12\Documents\vpc\speech1.wav'

# Load the audio file manually
waveform, sample_rate = librosa.load(sample1, sr=None)  # Don't resample initially

# Convert to mono if stereo
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)  # Convert to mono by averaging the channels

# Resample to 16kHz if necessary
if sample_rate != 16000:
    waveform = librosa.resample(waveform, sample_rate, 16000)
    sample_rate = 16000

# Convert the waveform to the required format for the processor
# This will create the necessary input features for the model
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

# Use the pipeline to transcribe the audio
# Now we access the input_features key and convert it to numpy
input_features = inputs["input_features"].numpy()
print(input_features.shape)
# Pass the input_features as NumPy array to the pipeline
result = pipe(input_features, return_timestamps=True)

# Print the transcription result
print(result["text"])




# wer_metric = load("wer")

# wer = wer_metric.compute(references=[reference], predictions=[prediction])

# print(wer)