
#IAV, 특징벡터 찾기, 음성구간 추출, 음성구간만 이어서 wav파일로 다시 저장, 원래 data랑 추출된 data 비교(근데 scale 안맞음 참고)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def IAV(audio_file, output_file):
    # 음성 로드
    waveform, sample_rate = sf.read(audio_file)

    #sampling rate 출력 16k or 44.1k
    #print("Sampling Rate:", sample_rate)

    # 음성을 프레임으로 분할
    frame_length = int(0.025 * sample_rate)  # 프레임 길이 (샘플 수)
    frame_hop = int(0.010 * sample_rate)  # 프레임 간격 (샘플 수)

    # 프레임 인덱스 계산
    num_frames = (len(waveform) - frame_length) // frame_hop + 1
    frame_indices = [(i * frame_hop, i * frame_hop + frame_length) for i in range(num_frames)]

    # 프레임 추출
    frames = [waveform[start:end] for start, end in frame_indices]

    # IAV 특징 벡터 추출
    iav_features = np.sum(np.abs(frames), axis=1) / frame_length

    # IAV 특징 벡터 출력
    #print(iav_features)

    # 최댓값과 최솟값 계산
    max_value = np.max(iav_features)
    min_value = np.min(iav_features)

    # 최솟값 설정
    min_threshold = (max_value - min_value) * 0.1 + min_value
    if min_threshold > max_value * 0.7:
        min_threshold = max_value * 0.2

    # 음성 구간 추출
    speech_segments = []
    is_speech = False

    for i in range(len(iav_features)):
        if iav_features[i] > min_threshold:
            if not is_speech:
                start_index = frame_indices[i][0]
                is_speech = True
        else:
            if is_speech:
                end_index = frame_indices[i-1][1]
                is_speech = False
                speech_segments.append((start_index, end_index))

    # 추출된 음성 구간 출력
    for segment in speech_segments:
        print("음성 구간의 시작 인덱스:", segment[0])
        print("음성 구간의 끝 인덱스:", segment[1])

    # 추출된 음성 구간을 하나의 배열로 연결
    concatenated_audio = np.concatenate([waveform[start:end] for start, end in speech_segments])

    # 새로운 WAV 파일로 저장
    sf.write(output_file, concatenated_audio, sample_rate)

    print("음성 구간이 추출되어", output_file, "파일로 저장되었습니다.")

    return concatenated_audio

    #그래프 표시
    #plt.plot(waveform, color='blue', label='Original Audio')
    #plt.plot(concatenated_audio, color='red', label='Extracted Speech Segments')

    #plt.xlabel("Time")
    #plt.ylabel("Amplitude")
    #plt.title("Comparison of Original Audio and Extracted Speech Segments")
    #plt.legend()
    #plt.show()

#결국 여기선 IAV threshold를 적용한 음성 특징벡터를 추출함, 음성구간만을 추출, 3.3으로 계속

