from json_as_a_judge.config import HF_CACHE_PATH

import torchaudio
from speechbrain.inference import EncoderClassifier

speechbrain_classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa", 
        savedir=HF_CACHE_PATH + "/models--sb-accent" 
    )

def get_accent(audio_path):
    """
    - Speechbrain classifier from this repo - https://github.com/JuanPZuluaga/accent-recog-slt2022 
    - Outputs cosine sim scores between class embeddings and audio embeddings
    """

    cos_sim_scores, score, index, text_lab = speechbrain_classifier.classify_file(audio_path)

    accent_labels = [
        'england', 'us', 'canada', 'australia', 'indian', 'scotland', 'ireland',
        'african', 'malaysia', 'newzealand', 'southatlandtic', 'bermuda',
        'philippines', 'hongkong', 'wales', 'singapore'
    ]

    sim_dict = {label: round(float(cos_sim_scores[0][i]), 3) for i, label in enumerate(accent_labels)}

    return sim_dict
