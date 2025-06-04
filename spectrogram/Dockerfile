# Use a base image with Python and a recent Linux distribution
# debian:bookworm-slim is a good choice for smaller image size and includes Python 3.11
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for librosa/pydub and ffmpeg
# - ffmpeg: Essential for MP3 decoding
# - build-essential: For compiling some Python packages if needed (optional but good practice)
# - libsndfile1: Often a dependency for sound libraries
# - libportaudio2: For audio I/O if you were doing live recording in the container (not directly needed for file processing but can resolve some audio errors)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Reduces image size by not caching pip packages
# --upgrade pip: Ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your Python application code into the container
COPY audio_spectrogram.py .
# Optionally copy an example MP3 for demonstration/testing inside the container
COPY test_audio.mp3 .

# Set the entry point for the container.
# This means when the container runs, it will execute your Python script.
# The `python -u` flag ensures unbuffered output, which is good for Docker logs.
ENTRYPOINT ["python", "-u", "audio_spectrogram.py"]

# Command to be executed by the entrypoint.
# This will be passed as arguments to `python -u audio_spectrogram.py`
# If you don't provide any arguments when running `docker run`, this will be used.
# If you provide arguments (e.g., `docker run my-spectrogram-app another_audio.mp3`),
# those arguments will override this CMD.
CMD ["test_audio.mp3"]
