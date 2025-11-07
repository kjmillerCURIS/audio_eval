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
    - word_timestamps: list of [(text, start, end)]
    """

    def _format_word_chunks(word_chunks):
        word_tuples = [
            (chunk["text"].strip(), chunk["timestamp"][0], chunk["timestamp"][1])
            for chunk in word_chunks
        ]

        return word_tuples

    transcription = whisper_pipe(
        audio_path,
        return_timestamps="word",
        generate_kwargs={"language": "english"}
    )

    transcript = transcription["text"]
    word_chunks = transcription["chunks"]

    return transcript, _format_word_chunks(word_chunks)




def whisperx_transcribe(audio_path: str):
    """
    - Same as whisper transcribe but less accurate for word level timestamps
    - Whisperx transcription model has environment issues 
    - Word level timestamps are off
    - Use audio_data/speakbench508_audio/42/audio_a.wav to see failure case
    """

    import whisperx

    def _format_whisperx_word_ts(word_list):
        """
        Formats output to be same as HF whisper model
        """
        formatted = []
        for w in word_list:
            formatted.append({
                "text": w["word"],
                "timestamp": (float(w["start"]), float(w["end"]))
            })
        return formatted

    
    whisperx_model = whisperx.load_model(
        "large-v3", 
        device="cuda", 
        compute_type="float16", 
        download_root=HF_CACHE_PATH
    )

    audio = whisperx.load_audio(audio_path)
    result = whisperx_model.transcribe(audio, batch_size=1)
    print(result["segments"]) # before alignment

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda", model_dir=HF_CACHE_PATH)
    result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda", return_char_alignments=False)

    print(result["segments"]) # after alignment
