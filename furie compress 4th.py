import numpy as np
from scipy.fftpack import fft, ifft

def compress_data(signal, threshold=0.01):
    """データをフーリエ変換して圧縮する"""
    signal_fft = fft(signal)
    half_size = len(signal) // 2
    signal_fft_half = signal_fft[:half_size]

    amplitude_half = np.abs(signal_fft_half)
    phase_half = np.angle(signal_fft_half)
    frequencies = np.fft.fftfreq(len(signal), d=t[1]-t[0])[:half_size]

    significant_indices = amplitude_half > threshold
    compressed_data = np.vstack((frequencies[significant_indices], amplitude_half[significant_indices], phase_half[significant_indices])).T

    return compressed_data.astype(np.float32).tobytes()

def decompress_data(compressed_data, original_size):
    """圧縮データを解凍して元の形式に戻す"""
    compressed_array = np.frombuffer(compressed_data, dtype=np.float32).reshape(-1, 3)
    loaded_frequencies = compressed_array[:, 0]
    loaded_amplitudes = compressed_array[:, 1]
    loaded_phases = compressed_array[:, 2]

    half_size = original_size // 2
    restored_fft_half = np.zeros(half_size, dtype=np.complex128)
    significant_indices = loaded_frequencies > 0
    restored_fft_half[significant_indices] = loaded_amplitudes * np.exp(1j * loaded_phases)

    restored_fft = np.zeros(original_size, dtype=np.complex128)
    restored_fft[:half_size] = restored_fft_half
    restored_fft[half_size:] = np.conj(restored_fft_half[::-1])

    restored_signal = ifft(restored_fft)
    return restored_signal.real

# サンプルデータの作成
np.random.seed(0)
t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.1, len(t))

# データの圧縮
compressed_data = compress_data(signal)

# 圧縮データのサイズ
compressed_size = len(compressed_data)

# 元のデータサイズ
original_size = signal.nbytes

# 圧縮比率の計算
compression_ratio = original_size / compressed_size
compression_percentage = (1 - compressed_size / original_size) * 100

# データの解凍
restored_signal = decompress_data(compressed_data, len(signal))

# 結果の表示
print("Compression Ratio:", compression_ratio)
print("Compression Percentage:", compression_percentage)
print("Original Size (bytes):", original_size)
print("Compressed Size (bytes):", compressed_size)