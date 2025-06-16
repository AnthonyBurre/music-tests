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
    # libsndfile1 \
    # libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Explicitly create the group first (optional, adduser usually does this)
#    but being explicit ensures it's there.
RUN groupadd --system appgroup
# Create the user, associating them with the created group
#    `--ingroup appgroup` ensures the user's primary group is 'appgroup'
RUN adduser --system --ingroup appgroup --home /app appuser

# Ensure necessary directories are writable by the non-root user
# For example, if your app needs to write to a log directory or an output directory
# RUN mkdir -p /app/output && chown appuser:appgroup /app/output

RUN chown appuser:appgroup /app


# Copy the requirements file first to leverage Docker's build cache
COPY app/requirements.txt .

# Install Python dependencies
# --no-cache-dir: Reduces image size by not caching pip packages
# --upgrade pip: Ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your Python application code into the container
COPY app/audio_spectrogram.py .

# Switch to the non-root user for runtime
USER appuser

# Set the entry point for the container.
# The `python -u` flag ensures unbuffered output, which is good for Docker logs.
ENTRYPOINT ["python", "-u", "audio_spectrogram.py"]

# Command to be executed by the entrypoint.
# This will be passed as arguments to `python -u audio_spectrogram.py`
# If you don't provide any arguments when running `docker run`, this will be used.
CMD ["--input_dir", "/app/input", "--output_dir", "/app/output"]
