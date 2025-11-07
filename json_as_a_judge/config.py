import os

HF_CACHE_PATH = "/projectnb/ivc-ml/nivek/audio_eval/json_as_a_judge/models"

os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_PATH
os.environ["HF_HOME"] = HF_CACHE_PATH

