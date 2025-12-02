import os
import sys
import json
from tqdm import tqdm
sys.path.append('.')
from WinoSound.tts_generator import TTSGenerator, client
from WinoSound.compute_baseline_audio_stats import compute_stats_one_audio


NUM_BASELINE_REPS = 5
OVERRIDE_THRESHOLD = 3
DEFAULT_TEXT = 'You are hearing me articulate because this is how I normally sound.'


def handle_baseline(text, target_emotion, voice, name, attempt_dir, offset):
    if target_emotion not in ['hesitant', 'frazzled', 'impatient', 'empathetic']:
        return

    stats = []
    my_generator = TTSGenerator()
    for rep in tqdm(range(NUM_BASELINE_REPS)):
        neutral_audio_path = os.path.join(attempt_dir, '%s-neutralrep%02d.wav'%(name, rep))
        if target_emotion == 'hesitant':
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=DEFAULT_TEXT,
            ) as response:
                response.stream_to_file(neutral_audio_path)
        else:
            overrides = 0
            while True:
                success = my_generator.generate(text, 'neutral', voice, neutral_audio_path, offset)
                if success:
                    break

                overrides += 1
                if overrides >= OVERRIDE_THRESHOLD:
                    print('enough attempts! just do without template!')
                    success = my_generator.generate(text, 'neutral', voice, neutral_audio_path, offset, override_template=True)
                    assert(success)
                    break

        stats_one = compute_stats_one_audio(neutral_audio_path)
        stats.append(stats_one)

    stats_filename = os.path.join(attempt_dir, '%s-neutral-stats.json'%(name))
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=4)
