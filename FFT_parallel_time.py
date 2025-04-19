import numpy as np
import time
import matplotlib.pyplot as plt
import gc
from multiprocessing import Pool

# === Parameters ===
N = 2**18  # 65,536 samples (safe for memory)
wave_counts = [10, 100, 1000, 10000]
num_chunks = 4  # Fixed number of processes (cores)

def generate_signal(N, num_waves):
    t = np.linspace(0, 1, N)
    signal = np.zeros(N)
    for i in range(1, num_waves + 1):
        amp = 1 / i  # Decreasing amplitude
        signal += amp * np.sin(2 * np.pi * i * 5 * t)
    signal += 0.3 * np.random.randn(N)
    return signal

def compute_fft_parallel(signal, num_chunks):
    chunk_size = len(signal) // num_chunks
    chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    with Pool(processes=num_chunks) as pool:
        fft_chunks = pool.map(np.fft.fft, chunks)
    gc.collect()
    return np.concatenate(fft_chunks)

# === Run and Record Execution Times ===
parallel_times = []

if __name__ == "__main__":
    print(f"Running Parallel FFT with {num_chunks} cores...\n")

    for count in wave_counts:
        signal = generate_signal(N, count)

        start = time.perf_counter()
        _ = compute_fft_parallel(signal, num_chunks)
        end = time.perf_counter()

        elapsed = end - start
        parallel_times.append(elapsed)
        print(f"{count} waveforms → {elapsed:.6f} sec")

    # === Plotting ===
    plt.figure(figsize=(8, 5))
    plt.plot(wave_counts, parallel_times, marker='o', linestyle='-', color='green')
    plt.title(f"Parallel FFT Execution Time (Cores = {num_chunks})")
    plt.xlabel("Number of Waveforms")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
