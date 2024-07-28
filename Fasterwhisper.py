# -*- coding: UTF-8 -*-
import torch
import whisperx
import numpy as np
from pydub import AudioSegment
from loguru import logger
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#from common.error import ErrorCode
#from server.asr.service.pyannote_whisper import diarize_text
#from pyannote_whisper.utils import diarize_text

from pyannote.core import Segment, Annotation, Timeline

def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed

class FastWhisperService():
    def __init__(self, config, options):
        self.config = config
        self.options = options

        self.model_path = "large-v3"
        self.model = None
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

    def load(self):
        logger.info(f"load: {self.model_path} on {self.device}, compute_type: {self.compute_type}")
        self.model = WhisperModel(self.model_path, device=self.device, compute_type=self.compute_type)

        # self.model = whisper.load_model("large-v3", device="cuda")

        logger.info(f"加载说话者识别模型...")
        self.spk_rec_pipeline = Pipeline.from_pretrained(r"C:\Users\ASUS\Desktop\zrt\chatbot\src\data\models\speaker-diarization-3.1\config.yaml")
        # self.spk_rec_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
        self.spk_rec_pipeline.to(torch.device(self.device))

    def get_asr_result(self, audio):
        try:
            # asr
            asr_result, info = self.model.transcribe(audio, language="zh", beam_size=5)

            # 说话者识别
            diarization_result = self.spk_rec_pipeline(audio)

            # language = info.language
            # language_probability = info.language_probability

            final_result = diarize_text(asr_result, diarization_result)
        except Exception as e:
            logger.error(f"fastwhisper Error: {e}")
            return {
                #"err": ErrorCode.FAILED.name.lower(),
                "result": None
            }

        result = []
        for segment, spk, sent in final_result:
            logger.info("[%.2fs -> %.2fs] %s %s" % (segment.start, segment.end, sent, spk))

            meta = {
                "text": sent,
                "start": segment.start,
                "end": segment.end,
                "spk": int(spk.split("_")[-1])
            }
            result.append(meta)

        return {
            #"err": ErrorCode.SUCCESS.name.lower(),
            'result': result
        }


class FastWhisperXServiceSingleton:
    _instance = None

    def __new__(cls, config, options):
        if cls._instance is None:
            cls._instance = super(FastWhisperXServiceSingleton, cls).__new__(cls)
            cls._instance.init(config, options)
        return cls._instance

    def init(self, config, options):
        self.config = config
        self.options = options

        self.whisper_model_path = "large-v3"
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

        self.align_model_path = r"C:\Users\ASUS\Desktop\meetalkcode\models\wav2vec2-large-xlsr-53-chinese-zh-cn"
        self.diarize_model_path = r"C:\Users\ASUS\Desktop\meetalkcode\models\speaker-diarization-3.1\config.yaml"
        self.language = None
        self.load()

    def load(self):
        logger.info(f"load: {self.whisper_model_path} on {self.device}, compute_type: {self.compute_type}")
        self.whisper_model = whisperx.load_model(self.whisper_model_path, self.device, compute_type=self.compute_type)
        self.diarize_model = whisperx.DiarizationPipeline(model_name=self.diarize_model_path, device=self.device)

        self.punc_model = pipeline(task=Tasks.punctuation,
                                   model='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
                                   model_revision="v2.0.4")

    def get_asr_result(self, audio_path):
        try:
            audio = whisperx.load_audio(audio_path)

            if self.language is None:
                asr_result = self.whisper_model.transcribe(audio, batch_size=2)
                self.language = asr_result["language"]
            else:
                asr_result = self.whisper_model.transcribe(audio, batch_size=2, language=self.language)

            model_a, metadata = whisperx.load_align_model(language_code=self.language, device=self.device,
                                                          model_name=self.align_model_path)
            align_result = whisperx.align(asr_result["segments"], model_a, metadata, audio, self.device,
                                          return_char_alignments=False)

            diarize_segments = self.diarize_model(audio)

            final_result = whisperx.assign_word_speakers(diarize_segments, align_result)

            result = []
            for item in final_result["segments"]:
                meta = {
                    "text": self.punc_model(item["text"])[0]['text'],
                    "start": round(item["start"], 1),
                    "end": round(item["end"], 1),
                    "spk": int(item["speaker"].split("_")[-1])
                }

                result.append(meta)

            return {
                #"err": ErrorCode.SUCCESS.name.lower(),
                "result": result
            }

        except Exception as e:
            logger.error(f"fastwhisperx asr Error: {e}")
            return {
                #"err": ErrorCode.FAILED.name.lower(),
                "result": None
            }


def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = ['.', '?', '!']


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_to_txt(spk_sent, file):
    with open(file, 'w') as fp:
        for seg, spk, sentence in spk_sent:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
            fp.write(line)



def run(audio):
    config = {
    }
    options = {}

    fast_whisper_x_service = FastWhisperXServiceSingleton(config, options)

    # Call the get_asr_result method with the audio file path
    result = fast_whisper_x_service.get_asr_result(audio)
    return result

print(run("output25.0.wav"))
print("25done")
print(run("output55.0.wav"))
print("55done")

