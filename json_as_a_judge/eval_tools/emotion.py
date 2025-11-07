from json_as_a_judge.config import HF_CACHE_PATH

from funasr import AutoModel

"""
- Can also conider adding: https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP for 4 way classifer 
"""

e2vec_model = AutoModel(
    model="iic/emotion2vec_plus_large",
    hub="hf",  
    cache_dir=HF_CACHE_PATH,
)

def emotion_to_vec_scores(audio_path):

    # Run inference
    rec_result = e2vec_model.generate(
        audio_path,
        granularity="utterance",
        extract_embedding=False,
    )

    keys = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

    scores = rec_result[0]['scores']
    result_dict = {k: round(s, 3) for k, s in zip(keys, scores)}

    return result_dict
