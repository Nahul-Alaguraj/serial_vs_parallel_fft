import numpy as np
import time
import matplotlib.pyplot as plt
import gc
from multiprocessing import Pool

# === Parameters ===
N = 2**16
num_waveforms = 100000  # Constant
core_counts = [1, 2, 4]  # Don't go beyond 4 for safety
timings = []

def generate_signal(N, num_waves):
    t = np.linspace(0, 1, N)
    signal = np.zeros(N)
    for i in range(1, num_waves + 1):
        amplitude = 1 / i
        freq = i * 5
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    signal += 0.3 * np.random.randn(N)
    return signal

def compute_fft_parallel(signal, num_chunks):
    chunk_size = len(signal) // num_chunks
    chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    with Pool(processes=num_chunks) as pool:
        fft_chunks = pool.map(np.fft.fft, chunks)
    gc.collect()
    return np.concatenate(fft_chunks)

if __name__ == "__main__":
    print(f"Running Parallel FFT Scalability Test for {num_waveforms} waveforms...\n")

    signal = generate_signal(N, num_waveforms)

    for cores in core_counts:
        print(f"Testing with {cores} core(s)...")
        start = time.perf_counter()
        _ = compute_fft_parallel(signal, cores)
        end = time.perf_counter()
        elapsed = end - start
        timings.append(elapsed)
        print(f"{cores} core(s) → {elapsed:.6f} seconds\n")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(core_counts, timings, marker='o', linestyle='-', color='purple')
    plt.title(f"Parallel FFT Execution Time vs Number of Cores (Waveforms = {num_waveforms})")
    plt.xlabel("Number of Cores")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
