from evaluation.config import HF_CACHE_PATH

from funasr import AutoModel

def emotion_to_vec_scores(audio_path, model_id = "iic/emotion2vec_plus_large"):

    model = AutoModel(
        model=model_id,
        hub="hf",  
        cache_dir=HF_CACHE_PATH,
    )

    # Run inference
    rec_result = model.generate(
        audio_path,
        granularity="utterance",
        extract_embedding=False,
    )

    keys = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

    scores = rec_result[0]['scores']
    result_dict = {k: round(s, 3) for k, s in zip(keys, scores)}

    return result_dict