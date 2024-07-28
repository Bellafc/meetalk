# import whisper
# import soundfile
# import os
#
# chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
# encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
# decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention
#
# model = whisper.load_model("tiny")
#
# wav_file = r"C:\Users\ASUS\Desktop\meetalkcode\test.wav"
# speech, sample_rate = soundfile.read(wav_file)
# chunk_stride = chunk_size[1] * 960 # 600ms
#
# cache = {}
# total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)
# for i in range(total_chunk_num):
#     speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
#     is_final = i == total_chunk_num - 1
#     result = model.transcribe(audio=speech_chunk, language='zh-cn')
#     print(result['text'])

import requests
from pydub import AudioSegment
#import whisper
#model = whisper.load_model("large",device='cuda')
import json

audio = AudioSegment.from_file(r"C:\Users\ASUS\Desktop\meetalkcode\uploads\1121201_陳蘭心_諮詢錄音.mp3", format="mp3")

# 设置分割参数
segment_length=30000
segments=[]
print("duration",audio.duration_seconds)
for start_time in range(0, int(audio.duration_seconds)*10000,30000):
    if start_time>5000:
        start_time=start_time-5000
    else:
        start_time=start_time
    end_time=start_time+segment_length
    segment=audio[start_time:end_time]
    segments.append(segment)
    segment.export("output"+str(start_time/1000)+".wav",format='wav')
    print(segment,str(start_time))
    audionew=AudioSegment.from_file("output"+str(start_time/1000)+".wav")
    print(audionew.duration_seconds)

   # result = model.transcribe("output"+str(start_time/1000)+".wav")
   # print(result["text"])
    url = "http://117.50.188.175:14095/asr/get_asr_result?model=funasr"

    payload = {}
    files=[
    ('audio_file',("output"+str(start_time/1000)+".wav",open("output"+str(start_time/1000)+".wav",'rb'),'audio/wav'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # 初始化变量
    current_spk = None
    current_text = []

    print("response",response)
    print("response txt", response.text)
    data = json.loads(response.text)
    for item in data['result']:
        spk = item['spk']
        text = item['text']

        if spk != current_spk:
            if current_spk is not None:
                print(f"spk:{current_spk}, {''.join(current_text)}")
            current_spk = spk
            current_text = [text]
        else:
            current_text.append(text)
    if current_spk is not None:
        print(f"spk:{current_spk}, {''.join(current_text)}")


