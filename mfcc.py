# MFCC
import numpy as np
import math

# pre_emphasis
def pre_emphasis(signal, pre_emphasis_coeff):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])
    return emphasized_signal

# Hamming window
def windowing_signal(signal, window_size, overlap_ratio):
    # Overlapping 계산
    overlap_size = int(window_size * overlap_ratio)
    hop_size = window_size - overlap_size

    # 신호 길이 계산
    signal_length = len(signal)

    # 프레임 수 계산
    num_frames = 1 + int((signal_length - window_size) / hop_size) + 1
    #print(num_frames) #227

    # 유동적인 zero-padding
    pad_length = (num_frames - 1)  * hop_size + window_size - signal_length
    padded_signal = np.pad(signal, (0, pad_length), mode='constant')
    #print(np.size(padded_signal))

    # windowing된 프레임을 저장할 배열 초기화
    frames = np.zeros((num_frames, window_size))

    # Hamming Window 계수 계산
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))

    # 프레임 단위로 Hamming Window 적용
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        #print(start,end)
        #print(i)

        # 신호의 끝을 벗어나지 않도록 조정
        if end > signal_length:
            end = signal_length
            start = end - window_size

        # 현재 프레임에 Hamming Window 적용
        frames[i] = padded_signal[start:end] * hamming_window
        #print(np.size(frames[i]))

    return frames

# fft
def my_rfft(frames, fft_size):
    num_frames = frames.shape[0]
    frame_size = frames.shape[1]

    # Padding
    padded_frames = np.pad(frames, ((0, 0), (0, fft_size - frame_size)))

    # FFT 계산
    spectrum = np.zeros((num_frames, fft_size // 2 + 1), dtype=np.complex64)
    for i in range(num_frames):
        for k in range(fft_size // 2 + 1):
            real_sum = 0.0
            imag_sum = 0.0
            for n in range(frame_size):
                angle = 2 * np.pi * k * n / fft_size
                real_sum += padded_frames[i, n] * np.cos(angle)
                imag_sum -= padded_frames[i, n] * np.sin(angle)
            spectrum[i, k] = complex(real_sum, imag_sum)

    return spectrum

def get_mel_filterbank(sample_rate, fft_size, num_filters):
    min_freq = 0
    max_freq = sample_rate / 2
    mel_min = hz_to_mel(min_freq)
    mel_max = hz_to_mel(max_freq)
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_filters, fft_size // 2 + 1))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            filterbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            filterbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    return filterbank

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

# dct
def get_dct_matrix(num_filters, num_cepstral):
    dct_matrix = np.zeros((num_cepstral, num_filters))
    for i in range(num_cepstral):
        for j in range(num_filters):
            dct_matrix[i, j] = math.sqrt(2.0 / num_filters) * math.cos(
                (i + 0.5) * math.pi * j / num_filters
            )
    return dct_matrix


# mfcc
def get_mfcc(iav_signal, sample_rate, num_cepstral, num_filters, fft_size):
    
    window_size = 250
    overlap_ratio = 0.5
    pre_signal =pre_emphasis(iav_signal,0.95)
    win_signal = windowing_signal(pre_signal, window_size, overlap_ratio)
    #print(win_signal)

    # Magnitude Spectrum
    magnitude_spectrum = my_rfft(win_signal, fft_size)
    magnitude_spectrum = np.abs(magnitude_spectrum)

    # Mel Filterbank
    mel_filterbank = get_mel_filterbank(sample_rate, fft_size, num_filters)
    mel_spectrum = np.dot(magnitude_spectrum, mel_filterbank.T)

    # Logarithm
		#0으로 나누어지는 오류를 방지하기 위해
    mel_spectrum = np.log(mel_spectrum + 1e-10)

    # Discrete Cosine Transform (DCT)
    dct_matrix = get_dct_matrix(num_filters, num_cepstral)
    mfcc = np.dot(mel_spectrum, dct_matrix.T)
    
    return mfcc



#window_size = 250
#overlap_ratio = 0.5
#signal = np.zeros(1000)
#pre_signal = pre_emphasis(signal,0.95)
#print(windowing_signal(pre_signal,window_size,overlap_ratio))