#import whisper
#model = whisper.load_model("base",device='cuda')
#result = model.transcribe(r"C:\Users\ASUS\Desktop\meetalkcode\uploads\1121201_陳蘭心_諮詢錄音.mp3")
#print(result["text"])

#from funasr import AutoModel

#chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
#encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
#decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

#model = AutoModel(model="paraformer-zh-streaming")

#import soundfile
#import os

#wav_file = os.path.join(model.model_path, r"C:\Users\ASUS\Desktop\meetalkcode\testingaudios\1130408_楊宜蓉_錄音檔.mp3")
#print(wav_file)
#speech, sample_rate = soundfile.read(wav_file)
#print("speech.shape",speech.shape,"sample_rate", sample_rate)
#chunk_stride = chunk_size[1] * 960 # 600ms

#cache = {}
#total_chunk_num = int(len((speech)-1)/chunk_stride+1)
#for i in range(total_chunk_num):
#    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
#    print(speech_chunk.shape)
#    is_final = i == total_chunk_num - 1
#    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#    print(res)
from funasr import AutoModel
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
#
# import soundfile
# import os
#
# wav_file = os.path.join(model.model_path, r"C:\Users\ASUS\Desktop\meetalkcode\test.wav")
# speech, sample_rate = soundfile.read(wav_file)
# chunk_stride = chunk_size[1] * 96000  # 600ms
#
# cache = {}
# total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
# for i in range(total_chunk_num):
#   speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
#   is_final = i == total_chunk_num - 1
#   res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
#                        encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#   print(res[0]['text'])
#import requests

#url = "http://117.50.188.175:14095/asr/get_asr_result?model=funasr"

#payload = {}
#files=[
 # ('audio_file',('1130408_楊宜蓉_錄音檔.mp3',open(r"C:\Users\ASUS\Desktop\meetalkcode\testingaudios\1130408_楊宜蓉_錄音檔.mp3",'rb'),'audio/wav'))
#]
#headers = {}

#response = requests.request("POST", url, headers=headers, data=payload, files=files)

#print(response.text)

import sys
print(sys.path)