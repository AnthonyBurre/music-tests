import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import AudioSegment # For robust MP3 loading and potential conversion

def load_mp3_to_mono_wav_array(mp3_path, sr=22050):
    """
    Loads an MP3 file, converts it to mono WAV (if needed), and loads it as a
    NumPy array with a specified sample rate.

    Args:
        mp3_path (str): Path to the MP3 file.
        sr (int): Target sample rate. Defaults to 22050 Hz (common for audio ML).

    Returns:
        tuple: (audio_time_series, sample_rate) as (np.ndarray, int)
               Returns None, None if the file cannot be processed.
    """
    if not os.path.exists(mp3_path):
        print(f"Error: File not found at {mp3_path}")
        return None, None

    try:
        # pydub can load MP3 directly and handles internal conversion to a format librosa can read
        audio = AudioSegment.from_mp3(mp3_path)

        # Convert to mono if it's stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample and convert to a NumPy array
        # pydub's get_array_of_samples returns 16-bit integers, convert to float for librosa
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # If pydub's sample rate is different from target_sr, librosa.load will resample
        # This approach ensures librosa's robust resampling is used.
        
        # A more direct approach with librosa is to let librosa.load handle it,
        # but pydub can ensure ffmpeg is correctly invoked first.
        
        # Let's use librosa's direct loading, as it has built-in MP3 support via audioread/ffmpeg
        # This is usually the cleanest way.
        y, loaded_sr = librosa.load(mp3_path, sr=sr, mono=True)
        return y, loaded_sr

    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")
        return None, None


def main(input_dir, output_dir):
    """
    main function for file processing
    """
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}. Please ensure it exists and is mounted correctly.")
        exit(1)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            input_filepath = os.path.join(input_dir, filename)
            print(f"\nProcessing '{input_filepath}'...")

            # Load MP3
            audio_series, sr = load_mp3_to_mono_wav_array(input_filepath, sr=22050)

            if audio_series is not None:
                print(f"Audio loaded. Sample Rate: {sr} Hz, Duration: {len(audio_series)/sr:.2f} seconds")

            else:
                print("Failed to load audio. Cannot generate spectrograms.")


# Example Usage:
if __name__ == "__main__":

    parser = argparse.ArgumentParser("file options test script")

    parser.add_argument("--input_dir", type=str, default="./input",
                        help="Directory containing input MP3 files. Defaults to ./input.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save processed MP3 files and summary. Defaults to ./output.")
    
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    main(input_dir, output_dir)
    
