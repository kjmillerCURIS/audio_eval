from json_as_a_judge.config import HF_CACHE_PATH

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype="auto",
    device_map="auto",
)


def whisper_transcribe(audio_path: str):
    """
    Transcribes audio and returns:
    - transcript: full text
    - word_timestamps: list of {text, timestamp=(start, end)}
    """

    transcription = whisper_pipe(
        audio_path,
        return_timestamps="word",
        generate_kwargs={"language": "english"}
    )

    transcript = transcription["text"]
    word_chunks = transcription["chunks"]

    return transcript, word_chunks
