import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load audio
fs, signal = wavfile.read("The Chainsmokers - Closer (Lyric) ft. Halsey.wav")

# Convert to mono if needed
if len(signal.shape) == 2:
    left = signal[:, 0]
    right = signal[:, 1]
    signal = (left + right) / 2

# Convert the signal from int to float
signal = signal.astype(np.float32)

print("Sampling frequency:", fs)
print("Signal length:", len(signal))

start_time = 10
duration = 3
start = int(start_time * fs)
end = int((start_time + duration) * fs)
query_clip = signal[start:end]

print("Clip length:", len(query_clip))


# Plot Full Signal (Time Domain) - show only first 5 seconds
plt.figure()
portion = signal[:5 * fs]  # take only the first 5 seconds worth of samples
time_axis = np.linspace(0, 5, len(portion))  # time axis from 0 to 5 seconds
plt.plot(time_axis, portion)
plt.title("Full Signal (Time Domain) - First 5 Seconds")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Plot Clip (Time Domain)
plt.figure()
plt.plot(np.linspace(0, len(query_clip)/fs, len(query_clip)), query_clip)
plt.title("Clip (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# FFT function to get one-sided magnitude spectrum
def fft_magnitude(x):
    result = np.fft.fft(x)
    result = np.abs(result)
    half = len(result) // 2
    return result[:half]

clip_fft = fft_magnitude(query_clip)

# Create frequency axis
N = len(query_clip)
T = 1 / fs
freqs = np.fft.fftfreq(N, T)
freqs = freqs[:N//2]

# Plot Clip Frequency Spectrum
plt.figure()
plt.plot(freqs, clip_fft)
plt.title("Clip Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# Part 6: Perform Audio Matching
window_length = len(query_clip)
step_size = int(fs / 4)
similarity_scores = []
timestamps = []

for i in range(0, len(signal) - window_length, step_size):
    current_segment = signal[i : i + window_length]
    segment_fft = fft_magnitude(current_segment)

    dot_product = np.dot(clip_fft, segment_fft)
    score = dot_product / (np.linalg.norm(clip_fft) * np.linalg.norm(segment_fft))
    similarity_scores.append(score)
    timestamps.append(i / fs)

# Part 7: Detect the Best Match
best_match_idx = np.argmax(similarity_scores)
detected_time = timestamps[best_match_idx]  

# Part 8: Visualize Results
plt.figure(figsize=(10, 4))
plt.plot(timestamps, similarity_scores)
plt.title('Matching Similarity Score')
plt.xlabel('Time (seconds)')
plt.ylabel('Cosine Similarity')
plt.axvline(x=detected_time, color='r', linestyle='--', label='Detected Match')
plt.axvline(x=start_time, color='g', linestyle='--', label='Actual Clip Position')
plt.legend()
plt.show()

# Part 9: Compare Signals (Original vs Detected)
detected_segment = signal[int(detected_time * fs) : int(detected_time * fs) + window_length]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(query_clip)
plt.title("Original Query Clip")
plt.subplot(2, 1, 2)
plt.plot(detected_segment, color='orange')
plt.title("Detected Segment in Full Audio")
plt.tight_layout()
plt.show()

print("Original clip position:", start_time, "seconds")
print("Detected position:", detected_time, "seconds")
print("Best similarity score:", max(similarity_scores))