import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

# サンプル信号の作成（リニアデータ）
np.random.seed(0)
t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.1, len(t))

# フーリエ変換
signal_fft = fft(signal)

# フーリエ変換の対称性を利用するために信号の前半部分のみを使用
half_size = len(signal) // 2
signal_fft_half = signal_fft[:half_size]

# 振幅と位相
amplitude_half = np.abs(signal_fft_half)
phase_half = np.angle(signal_fft_half)
frequencies = np.fft.fftfreq(len(signal), d=t[1]-t[0])[:half_size]

# 主要な周波数成分を選択
threshold = 0.01
significant_indices = amplitude_half > threshold
significant_frequencies = frequencies[significant_indices]
significant_amplitudes = amplitude_half[significant_indices]
significant_phases = phase_half[significant_indices]

# xy関数として保存（例: numpy arrayを使って保存）
compressed_data = np.vstack((significant_frequencies, significant_amplitudes, significant_phases)).T

# バイナリ化
compressed_data_binary = compressed_data.astype(np.float32).tobytes()

# バイナリデータのサイズ
compressed_binary_size = len(compressed_data_binary)

# 元のデータと圧縮データのサイズを比較
original_size = signal.nbytes
compressed_size = compressed_data.nbytes
compression_ratio = original_size / compressed_size
compression_percentage = (1 - compressed_size / original_size) * 100

# バイナリ化後の圧縮比率と圧縮率の計算
compression_ratio_binary = original_size / compressed_binary_size
compression_percentage_binary = (1 - compressed_binary_size / original_size) * 100

# 結果のプロット
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(t, signal)
plt.subplot(3, 1, 2)
plt.title("Restored Signal")
plt.plot(t, ifft(signal_fft).real)
plt.subplot(3, 1, 3)
plt.title("Error (Original - Restored)")
plt.plot(t, signal - ifft(signal_fft).real)
plt.tight_layout()
plt.show()

compression_ratio_binary, compression_percentage_binary