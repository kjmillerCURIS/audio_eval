import io
import wave
from contextlib import contextmanager
from pydub import AudioSegment

@contextmanager
def open_audio_as_wave(path: str):
    audio = AudioSegment.from_file(path)  # mp3/wav/m4a/… -> decoded audio
    buf = io.BytesIO()
    audio.export(buf, format="wav")       # RIFF WAV (PCM) into memory
    buf.seek(0)
    wf = wave.open(buf, "rb")
    try:
        yield wf
    finally:
        wf.close()
        buf.close()

