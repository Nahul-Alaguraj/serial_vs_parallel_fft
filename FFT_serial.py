import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import os


num_waveforms = int(input("Enter number of waveforms (e.g., 10, 100, 1000): "))
amplitude_mode = input("Enter amplitude mode [equal / decreasing / random]: ").strip().lower()


N = 2**18
t = np.linspace(0, 1, N)
signal = np.zeros(N)


for i in range(1, num_waveforms + 1):
    freq = i * 5

    if amplitude_mode == "equal":
        amplitude = 1.0
    elif amplitude_mode == "decreasing":
        amplitude = 1 / i
    elif amplitude_mode == "random":
        amplitude = np.random.rand()
    else:
        raise ValueError("Invalid mode. Choose from: equal, decreasing, random")

    signal += amplitude * np.sin(2 * np.pi * freq * t)

# Add some noise
signal += 0.3 * np.random.randn(N)

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)
cpu_before = process.cpu_percent(interval=None)


start = time.perf_counter()
fft_result = np.fft.fft(signal)
end = time.perf_counter()

mem_after = process.memory_info().rss / (1024 * 1024)
cpu_after = process.cpu_percent(interval=None)


frequencies = np.fft.fftfreq(N, d=t[1] - t[0])
magnitude = np.abs(fft_result)
positive_freqs = frequencies[:N // 2]
positive_magnitude = magnitude[:N // 2]

print(f"\n[FFT Analysis with {num_waveforms} waveforms | Mode: {amplitude_mode}]")
print(f"FFT Time: {end - start:.6f} seconds")
print(f"Memory Used: {mem_after - mem_before:.2f} MB")
print(f"CPU Usage Delta: {cpu_after - cpu_before:.2f}%")


plt.figure(figsize=(12, 6))
plt.plot(positive_freqs, positive_magnitude)
plt.title(f"Magnitude Spectrum ({amplitude_mode.capitalize()} Amplitude, {num_waveforms} Waves)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(0, 2000)
plt.tight_layout()
plt.show()
