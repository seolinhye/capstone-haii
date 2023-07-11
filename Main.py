from IAV import IAV
from mfcc import get_mfcc
import numpy as np
import sys
import librosa 
import matplotlib.pyplot as plt

audio_file = "happy2.wav"
output_file = "After_IAV.wav"

#preprocessing: IAV
#음성신호만 모은 배열
Iav_sig = IAV(audio_file, output_file) 
mfcc = get_mfcc(Iav_sig, 16000,  13,  26, 512)
np.set_printoptions(threshold=sys.maxsize)
print(mfcc)

# MFCC 그래프 그리기
# plt.figure(figsize=(10, 6))
# plt.imshow(mfcc, cmap='hot', origin='lower', aspect='auto')
# plt.colorbar(format='%+2.0f dB')
# plt.xlabel('Cepstral Coefficients')
# plt.ylabel('Frame')
# plt.title('MFCC')
# plt.show()


# def extract_mfcc_features(filename, sampling_rate, num_frames):
#     # 음성 신호 로드
#     signal, sr = librosa.load(filename, sr=sampling_rate)

#     # 프레임 단위로 신호를 분할
#     frames = librosa.util.frame(signal, frame_length=250, hop_length=125).T

#     # 유동적인 zero padding 계산
#     pad_length = num_frames - frames.shape[0]
#     padded_frames = np.pad(frames, ((0, pad_length), (0, 0)))

#     # MFCC 특징 벡터 추출
#     mfcc = librosa.feature.mfcc(y=padded_frames, sr=sr, n_mfcc=13)

#     return mfcc

# # 사용 예시
# filename = "After_IAV.wav"
# sampling_rate = 16000
# num_frames = 227

# mfcc_features = extract_mfcc_features(filename, sampling_rate, num_frames)
# print(mfcc_features)  # (13, 227)


# import librosa
# import librosa.display
# import matplotlib.pyplot as plt

# #Iav_sig는 IAV 함수를 통해 얻은 신호
# Iav_sig = IAV(audio_file, output_file) 

# #MFCC 추출
# mfcc = get_mfcc(Iav_sig, 16000, 13, 26, 512)

# # MFCC 그래프 플롯
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(mfcc), x_axis='time', sr=16000, hop_length=256)
# plt.colorbar(format='%+2.0f dB')
# plt.xlabel('Time')
# plt.ylabel('MFCC Coefficients')
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()


# import librosa
# import librosa.display
# import matplotlib.pyplot as plt

# # Iav_sig는 IAV 함수를 통해 얻은 신호
# Iav_sig = IAV(audio_file, output_file) 

# # MFCC 추출
# mfcc = librosa.feature.mfcc(y=Iav_sig, sr=16000, n_mfcc=13)

# # MFCC 그래프 플롯
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(mfcc), x_axis='time', sr=16000, hop_length=256)
# plt.colorbar(format='%+2.0f dB')
# plt.xlabel('Time')
# plt.ylabel('MFCC Coefficients')
# plt.title('librosa MFCC')
# plt.tight_layout()
# plt.show()
