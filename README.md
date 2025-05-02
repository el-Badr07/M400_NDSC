# M400_NDSC

## Project Description

This repository is dedicated to speech recognition and text-to-speech processing using advanced models like Whisper and SpeechT5. The project aims to provide tools and scripts for automatic speech recognition (ASR) and text-to-speech (TTS) conversion, leveraging state-of-the-art models to achieve high accuracy and natural-sounding speech synthesis.

## Features

- Automatic Speech Recognition (ASR) using Whisper model
- Text-to-Speech (TTS) conversion using SpeechT5 model
- Batch processing of audio files for transcription
- Word Error Rate (WER) computation for evaluating ASR performance
- Support for multiple languages and voices

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/el-Badr07/M400_NDSC.git
   cd M400_NDSC
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Scripts

1. **main.py**: This script performs automatic speech recognition using the Whisper model.
   ```bash
   python main.py
   ```

2. **t5.py**: This script performs text-to-speech conversion using the SpeechT5 model.
   ```bash
   python t5.py
   ```

3. **test.py**: This script demonstrates the usage of the Kokoro TTS model for generating speech from text.
   ```bash
   python test.py
   ```

4. **test1.py**: This script saves an example speech audio from a dataset to a file.
   ```bash
   python test1.py
   ```

5. **testbatch.py**: This script performs batch processing of audio files for transcription and computes WER.
   ```bash
   python testbatch.py
   ```

6. **tt.py**: This script demonstrates handling stereo audio files and performing ASR using the Whisper model.
   ```bash
   python tt.py
   ```

7. **tt1.py**: This script performs ASR on multiple audio files and computes WER for the transcriptions.
   ```bash
   python tt1.py
   ```

8. **wer.py**: This script computes the Word Error Rate (WER) for given reference and prediction texts.
   ```bash
   python wer.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
