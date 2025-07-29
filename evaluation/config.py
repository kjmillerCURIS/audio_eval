import os

HF_CACHE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/models"

os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_PATH
os.environ["HF_HOME"] = HF_CACHE_PATH

