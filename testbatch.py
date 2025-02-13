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
def compute_wer(original_texts: list, transcribed_texts: list):
    from evaluate import load
    wer_metric = load("wer")

    wer_scores = []

    # Ensure both lists have the same length
    if len(original_texts) != len(transcribed_texts):
        raise ValueError("The number of original texts and transcribed texts must be the same.")

    # Loop over each pair of original and transcribed texts
    for orig_text, trans_text in zip(original_texts, transcribed_texts):
        # Compute WER for each pair
        wer_score = wer_metric.compute(references=[orig_text.lower()], predictions=[trans_text.lower()])
        wer_scores.append(wer_score)

    # Print out the WER for each pair
    for i, score in enumerate(wer_scores):
        print(f"WER for text pair {i + 1}: {score}")

    return wer_scores



# Specify the directory path
directory_path = r"C:\Users\PC-12\Documents\vpc\284449"
#directory_path=directory_path[:15]
# Use list comprehension to get all .flac files in the directory
multiple_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".flac")]
#multiple_files=multiple_files[:2]
# Now multiple_files will contain all the .flac file paths in the directory
print(multiple_files)
transcription_multiple = transcribe_audio(multiple_files)
print("Multiple files transcription results:", transcription_multiple)

# Example to iterate and process results
# for file, text in transcription_multiple.items():
#     if text: # Check if transcription was successful for the file
#         print(f"Transcription for {file}: {text}")
#     else:
#         print(f"Transcription failed for {file}")


texts = list(transcription_multiple.values())


def anonymize_text(text:list):
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


def anonymize_texts(texts: list):
    from kokoro import KPipeline
    from IPython.display import display, Audio
    import soundfile as sf

    # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
    # 🇯🇵 'j' => Japanese: pip install misaki[ja]
    # 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
    pipeline = KPipeline(lang_code='a')  # Ensure lang_code matches voice

    # Generate, display, and save audio files for each text in the list
    audios = []
    for idx, text in enumerate(texts):
        # Generate audio for each text
        generator = pipeline([text], voice='af_heart', speed=1, split_pattern=r'\n+')
        
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Index: {idx}, Audio {i}")
            print(f"Graphemes: {gs}")
            print(f"Phonemes: {ps}")
            
            # Display audio
            display(Audio(data=audio, rate=26000, autoplay=(i==0)))
            
            # Save the audio file
            audio_filename = f'{idx}_{i}.wav'
            audios.append(audio_filename)
            sf.write(audio_filename, audio, 26000)

    # After all texts are processed, transcribe and compute WER
    tc = transcribe_audio(audios)
    
    # Assuming that the `transcribe_audio` function returns a dict of transcriptions
    for i, text in enumerate(texts):
        # Retrieve transcribed texts (you should adjust the transcriptions based on how `transcribe_audio` works)
        #text_transcribed = tc.get(f"text_{i}", "")
        text_transcribed=list(tc.values())[i]
        print(f"Original: {text}\nTranscribed: {text_transcribed}")
        
        # Compute WER for the original and transcribed texts
        compute_wer([text], [text_transcribed])

# Call the function with a list of texts
texts = list(transcription_multiple.values())  # Replace with actual list of texts
anonymize_texts(texts)



