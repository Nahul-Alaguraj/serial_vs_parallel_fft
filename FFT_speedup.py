import matplotlib.pyplot as plt

# === Timing data ===
waveforms = [10, 100, 1000, 10000]

# Serial time (1 core)
serial_times = [0.0086, 0.0057, 0.0055, 0.0054]

# Parallel times
times_2_cores = [0.4034, 0.3724, 0.3789, 0.3835]
times_4_cores = [0.7496, 0.6593, 0.6495, 0.7005]

# === Speedup Calculations ===
speedup_2 = [s / p for s, p in zip(serial_times, times_2_cores)]
speedup_4 = [s / p for s, p in zip(serial_times, times_4_cores)]

# === Efficiency Calculations ===
efficiency_2 = [sp / 2 * 100 for sp in speedup_2]
efficiency_4 = [sp / 4 * 100 for sp in speedup_4]

# === Plot Speedup ===
plt.figure(figsize=(10, 5))
plt.plot(waveforms, speedup_2, 'o-', label='2 Cores', color='green')
plt.plot(waveforms, speedup_4, 's-', label='4 Cores', color='purple')
plt.title("Speedup vs Number of Waveforms")
plt.xlabel("Number of Waveforms")
plt.ylabel("Speedup")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot Efficiency ===
plt.figure(figsize=(10, 5))
plt.plot(waveforms, efficiency_2, 'o-', label='2 Cores', color='green')
plt.plot(waveforms, efficiency_4, 's-', label='4 Cores', color='purple')
plt.title("Efficiency vs Number of Waveforms")
plt.xlabel("Number of Waveforms")
plt.ylabel("Efficiency (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
