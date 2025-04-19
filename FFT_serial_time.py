import numpy as np
import matplotlib.pyplot as plt
import time

# --- Parameters ---
N = 2**18  # Total samples
wave_counts = [10, 100, 1000, 10000]
serial_times = []

def generate_signal(N, num_waves):
    t = np.linspace(0, 1, N)
    signal = np.zeros(N)
    for i in range(1, num_waves + 1):
        signal += (1 / i) * np.sin(2 * np.pi * i * 50 * t)
    return signal

# --- Serial FFT Execution ---
for count in wave_counts:
    signal = generate_signal(N, count)

    start = time.perf_counter()
    _ = np.fft.fft(signal)
    end = time.perf_counter()

    elapsed = end - start
    serial_times.append(elapsed)
    print(f"[Serial] {count} waveforms → {elapsed:.6f} sec")

# --- Plot Serial Results ---
plt.figure(figsize=(8, 5))
plt.plot(wave_counts, serial_times, 'o-', color='blue')
plt.title("Serial FFT Execution Time")
plt.xlabel("Number of Waveforms")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.show()
