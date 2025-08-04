from evaluation.config import HF_CACHE_PATH

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def whisper_transcribe(audio_path: str, model_id: str = "openai/whisper-large-v3"):
    """
    Transcribes audio and returns:
    - transcript: full text
    - word_timestamps: list of {text, timestamp=(start, end)}
    """

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype="auto",
        device_map="auto",
    )

    transcription = pipe(
        audio_path,
        return_timestamps="word",
        generate_kwargs={"language": "english"}
    )

    transcript = transcription["text"]
    word_chunks = transcription["chunks"]

    return transcript, word_chunks
