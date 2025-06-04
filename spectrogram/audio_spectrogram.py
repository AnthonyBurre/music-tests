import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import AudioSegment # For robust MP3 loading and potential conversion

def load_mp3_to_mono_wav_array(mp3_path, sr=22050):
    """
    Loads an MP3 file, converts it to mono WAV (if needed), and 
loads it as a
    NumPy array with a specified sample rate.

    Args:
        mp3_path (str): Path to the MP3 file.
        sr (int): Target sample rate. Defaults to 22050 Hz (common 
for audio ML).

    Returns:
        tuple: (audio_time_series, sample_rate) as (np.ndarray, 
int)
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
        # Create a dummy temp WAV file for librosa to load.
        # This is a robust way to ensure librosa handles the resampling correctly.
        # A more memory-efficient way might involve passing the raw samples to librosa's
        # functionality if librosa supports raw float arrays directly for STFT/Mel.
        # However, librosa.load is designed for file paths.
        
        # A more direct approach with librosa is to let librosa.load handle it,
        # but pydub can ensure ffmpeg is correctly invoked first.
        
        # Let's use librosa's direct loading, as it has built-in MP3 support via audioread/ffmpeg
        # This is usually the cleanest way.
        y, loaded_sr = librosa.load(mp3_path, sr=sr, mono=True)
        return y, loaded_sr

    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")
        return None, None


def create_spectrogram(audio_time_series, sr, n_fft=2048, hop_length=512, type='magnitude', window='hann'):
    """
    Creates a spectrogram (magnitude or Mel) from an audio time 
series.

    Args:
        audio_time_series (np.ndarray): The audio time series.
        sr (int): Sample rate of the audio.
        n_fft (int): FFT window size. Defaults to 2048.
        hop_length (int): Number of samples between successive 
frames. Defaults to 512.
        type (str): Type of spectrogram ('magnitude', 'power', or 
'mel').
                    'magnitude' for |STFT|, 'power' for |STFT|^2, 
'mel' for Mel spectrogram.
        window (str): Windowing function for FFT (e.g., 'hann', 
'hamm', 'blackman').

    Returns:
        np.ndarray: The spectrogram.
        np.ndarray: Frequencies (for non-Mel spectrograms).
    """
    if audio_time_series is None or sr is None:
        return None, None

    if type == 'magnitude' or type == 'power':
        # Compute the Short-Time Fourier Transform (STFT)
        stft = librosa.stft(y=audio_time_series, n_fft=n_fft, hop_length=hop_length, window=window)
        
        # Compute magnitude or power spectrogram
        if type == 'magnitude':
            spectrogram = np.abs(stft)
        else: # type == 'power'
            spectrogram = np.abs(stft)**2

        # Get frequencies for plotting
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return spectrogram, freqs

    elif type == 'mel':
        # Compute Mel spectrogram
        # It's common to compute a power spectrogram first, then convert to Mel
        # librosa.feature.melspectrogram does this internally
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_time_series,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0 # Compute power spectrogram before Mel filters
        )
        # Convert to decibels for better visualization and common ML practice
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db, None # Mel spectrograms don't have a direct linear frequency axis for plotting

    else:
        raise ValueError("Invalid spectrogram type. Choose 'magnitude', 'power', or 'mel'.")


def plot_spectrogram(spectrogram, sr, hop_length, title="Spectrogram", type='magnitude', save_path=None, figsize=(10, 5)):
    """
    Plots and optionally saves a spectrogram.

    Args:
        spectrogram (np.ndarray): The spectrogram to plot.
        sr (int): Sample rate of the audio.
        hop_length (int): Hop length used to create the 
spectrogram.
        title (str): Title for the plot.
        type (str): Type of spectrogram ('magnitude', 'power', or 
'mel') for correct y-axis.
        save_path (str, optional): Path to save the plot image 
(e.g., 'spectrogram.png').
                                   If None, the plot is displayed.
        figsize (tuple): Figure size for the plot.
    """
    if spectrogram is None:
        print("No spectrogram data to plot.")
        return

    plt.figure(figsize=figsize)
    librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis='time',
        y_axis='mel' if type == 'mel' else ('log' if type == 'power' else 'hz'), # 'log' for power, 'hz' for magnitude
        hop_length=hop_length,
        cmap='viridis' # Good colormap for spectrograms
    )
    plt.colorbar(format='%+2.0f dB' if type == 'mel' else None) # dB for Mel, otherwise default
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
        print(f"Spectrogram saved to {save_path}")
    else:
        plt.show()

# Example Usage:
if __name__ == "__main__":
    # Create a dummy MP3 file for testing (you can replace this with your own)
    # This requires pydub to be able to export, which needs ffmpeg
    try:
        from pydub import AudioSegment
        print("Creating a dummy MP3 file for demonstration...")
        # Generate a simple 5-second sine wave audio
        sample_rate = 44100
        duration_ms = 5000
        frequency = 440  # Hz (A4 note)
        amplitude = 0.5 # between 0 and 1

        t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * frequency * t) * (2**15 - 1) # Scale for 16-bit int
        waveform = waveform.astype(np.int16)

        dummy_audio = AudioSegment(
            waveform.tobytes(),
            frame_rate=sample_rate,
            sample_width=waveform.dtype.itemsize,
            channels=1
        )
        dummy_mp3_path = "dummy_audio.mp3"
        dummy_audio.export(dummy_mp3_path, format="mp3")
        print(f"Dummy MP3 file created: {dummy_mp3_path}")

    except Exception as e:
        print(f"Could not create dummy MP3 file (might need ffmpeg or pydub): {e}")
        print("Please provide your own MP3 file for testing if dummy_audio.mp3 is not created.")
        dummy_mp3_path = None # Set to None if dummy creation fails

    #if dummy_mp3_path and os.path.exists(dummy_mp3_path):
    if False:
        input_mp3_file = dummy_mp3_path
    else:
        # Fallback if dummy file creation fails or if you want to use your own
        # Replace 'your_audio.mp3' with the actual path to your MP3 file
        input_mp3_file = "test_audio.mp3"
        print(f"Attempting to use: {input_mp3_file}. Please ensure this file exists.")


    if os.path.exists(input_mp3_file):
        print(f"\nProcessing '{input_mp3_file}'...")

        # Load MP3
        audio_series, sr = load_mp3_to_mono_wav_array(input_mp3_file, sr=22050)

        if audio_series is not None:
            print(f"Audio loaded. Sample Rate: {sr} Hz, Duration: {len(audio_series)/sr:.2f} seconds")

            # --- Create and Plot a Magnitude Spectrogram ---
            magnitude_spec, freqs = create_spectrogram(audio_series, sr, type='magnitude')
            if magnitude_spec is not None:
                print("Magnitude Spectrogram created. Shape:", magnitude_spec.shape)
                plot_spectrogram(
                    magnitude_spec,
                    sr,
                    hop_length=512,
                    title=f"Magnitude Spectrogram of {os.path.basename(input_mp3_file)}",
                    type='magnitude',
                    save_path="magnitude_spectrogram.png"
                )

            # --- Create and Plot a Mel Spectrogram (common for ML) ---
            mel_spec_db, _ = create_spectrogram(audio_series, sr, type='mel')
            if mel_spec_db is not None:
                print("Mel Spectrogram created. Shape:", mel_spec_db.shape)
                plot_spectrogram(
                    mel_spec_db,
                    sr,
                    hop_length=512,
                    title=f"Mel Spectrogram (dB) of {os.path.basename(input_mp3_file)}",
                    type='mel',
                    save_path="mel_spectrogram.png",
                    figsize=(12,6) # Adjust size for Mel
                )
        else:
            print("Failed to load audio. Cannot generate spectrograms.")
    else:
        print(f"Skipping processing because '{input_mp3_file}' does not exist.")

    # Clean up dummy file if it was created
    if 'dummy_mp3_path' in locals() and dummy_mp3_path and os.path.exists(dummy_mp3_path):
        try:
            os.remove(dummy_mp3_path)
            print(f"Cleaned up dummy MP3 file: {dummy_mp3_path}")
        except Exception as e:
            print(f"Error cleaning up dummy file: {e}")
