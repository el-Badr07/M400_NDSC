import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import os
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
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

def transcribe_audio(file_paths):
    """
    Transcribes one or more audio files using the Whisper model.

    Args:
        file_paths: A string (single file path) or a list of strings (multiple file paths).

    Returns:
        A dictionary where keys are file paths and values are the transcribed text,
        or None if an error occurs.  Prints error messages to the console.
    """

    results = {}

    if isinstance(file_paths, str):  # Handle single file path
        file_paths = [file_paths]

    for file_path in file_paths:
        try:
            audio_array, sample_rate = sf.read(file_path)

            if audio_array.ndim > 1:  # Stereo to mono conversion
                audio_array = audio_array.mean(axis=1)

            audio_data = {"array": audio_array, "sampling_rate": sample_rate}
            result = pipe(audio_data)
            results[file_path] = result["text"]
            print(f"Transcription for {file_path}: {result['text']}") # Print transcription as it's processed

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results[file_path] = None  # Store None to indicate failure for that file

    return results

def compute_wer(text1:str,text2:str,text3:str,text4:str):

    from evaluate import load
    wer_metric = load("wer")
    #wer_metric=load("eer")
    wer1 = wer_metric.compute(references=[text1.lower()], predictions=[text2.lower()])
    wer2 = wer_metric.compute(references=[text3.lower()], predictions=[text4.lower()])
    print(wer1,wer2)
    return wer1,wer2


# Example usage:
# single_file = "path/to/your/audio.wav"
# transcription_single = transcribe_audio(single_file)
# print("Single file transcription results:", transcription_single)


# multiple_files = [r"C:\Users\PC-12\Documents\vpc\speech.wav", r"C:\Users\PC-12\Documents\vpc\speech1.wav"]  # List of paths
multiple_files = [r"C:\Users\PC-12\Documents\vpc\293981\367-293981-0000.flac", r"C:\Users\PC-12\Documents\vpc\293981\367-293981-0001.flac"]  # List of paths

transcription_multiple = transcribe_audio(multiple_files)
print("Multiple files transcription results:", transcription_multiple)

# Example to iterate and process results
# for file, text in transcription_multiple.items():
#     if text: # Check if transcription was successful for the file
#         print(f"Transcription for {file}: {text}")
#     else:
#         print(f"Transcription failed for {file}")

text1, text2 = transcription_multiple.values()


def anonymize_text(text1:str,text2:str):
    from kokoro import KPipeline
    from IPython.display import display, Audio
    import soundfile as sf
    # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
    # 🇯🇵 'j' => Japanese: pip install misaki[ja]
    # 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
    pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice

    # This text is for demonstration purposes only, unseen during training

    # 4️⃣ Generate, display, and save audio files in a loop.
    generator = pipeline(
        [text1,text2], voice='af_heart', # <= change voice here
        speed=1, split_pattern=r'\n+')
    audios=[]
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        display(Audio(data=audio, rate=26000, autoplay=i==0))
        audios.append(f'{i}.wav')
        sf.write(f'{i}.wav', audio, 26000) # save each audio file
    tc=transcribe_audio(audios)
    text11,text22=tc.values()
    compute_wer(text1,text11,text2,text22)



anonymize_text(text1,text2)





# # Example using os.listdir to process all wav files in a directory
# directory = "path/to/your/audio/directory"
# wav_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
# transcription_directory = transcribe_audio(wav_files)
# print("Directory transcription results:", transcription_directory)

text1="i swear answered sancho"
text11="\"'I swear,' answered Sancha."

text2="I say so, continued Don Quixote, because I hate taking away anyone's good name."
text22="I say so, continued Don Quixote, because I hate taking away anyone's good name."