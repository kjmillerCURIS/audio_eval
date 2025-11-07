import os
import sys
from tqdm import tqdm
sys.path.append('.')
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


NUM_REPS = 10
VOICES = ['echo', 'alloy', 'ash']
BASELINE_TEXT = 'You are hearing me articulate because this is how I normally sound.'
OUT_DIR = 'WinoSound/baseline_audios'


client = OpenAI()


def run_tts(text, voice, audio_path):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(audio_path)


def make_baseline_audios():
    os.makedirs(OUT_DIR, exist_ok=True)
    for voice in VOICES:
        for t in tqdm(range(NUM_REPS)):
            audio_path = os.path.join(OUT_DIR, 'baseline-%s-%02d.wav'%(voice, t))
            run_tts(BASELINE_TEXT, voice, audio_path)


if __name__ == '__main__':
    make_baseline_audios()
