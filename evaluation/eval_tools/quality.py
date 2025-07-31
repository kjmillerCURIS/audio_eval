from evaluation.config import HF_CACHE_PATH
from evaluation.libs.DNSMOS.dnsmos_single import ComputeScore, SAMPLING_RATE

import utmosv2
from jiwer import wer

def utmosv2_score(audio_path):

    model = utmosv2.create_model(pretrained=True)
    mos = model.predict(input_path=audio_path)
    mos_str = f"{mos:.2f} / 5.00"

    return mos_str

def dnsmos_score(audio_path):
    """
    Model files - 
    # https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx
    # https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
    # https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx

    Score docs: https://github.com/microsoft/DNS-Challenge/issues/123
    """
    
    def load_dnsmos_models(
        model_path="evaluation/models/DNSMOS/sig_bak_ovr.onnx",
        p_model_path="evaluation/models/DNSMOS/p_sig_bak_ovr.onnx",
        p808_model_path="evaluation/models/DNSMOS/model_v8.onnx",
    ):
        return ComputeScore(model_path, p_model_path, p808_model_path)

    # Load model
    scorer = load_dnsmos_models()

    # Compute scores
    dnsmos_scores = scorer(audio_path, sampling_rate=SAMPLING_RATE)

    interpretable_scores = {
        "DNSMOS_Personalized_Signal_Quality": f"{dnsmos_scores['P_SIG']:.2f} / 5.00",
        "DNSMOS_Personalized_Background_Quality": f"{dnsmos_scores['P_BAK']:.2f} / 5.00",
        "DNSMOS_Personalized_Overall_Quality": f"{dnsmos_scores['P_OVRL']:.2f} / 5.00",
        "P808_Overall_Quality": f"{dnsmos_scores['P808_MOS']:.2f} / 5.00",
    }

    return interpretable_scores

def word_error_rate(generated_text, transcribed_text):
    return round(wer(generated_text, transcribed_text), 3)

def audio_quality_scores(audio_path, generated_text, transcription):
    utmos_score = utmosv2_score(audio_path)

    # Get DNSMOS personalized scores (dict with formatted strings)
    dnsmos_personalized_scores = dnsmos_score(audio_path)

    word_err_rate = word_error_rate(generated_text, transcription)

    # Combine results into a single dict
    combined_scores = {
        "UTMOSv2_Mean_Opinion_Score": utmos_score,
        **dnsmos_personalized_scores,
        "WER": word_err_rate
    }

    return combined_scores
