import os
import sys
import json
import shutil
from tqdm import tqdm


def tts_copier_one(name, voice, target_emotion):
    src_dir = 'WinoSound/tts_samples'
    dst_dir = 'WinoSound/tts_samples_best/%s/%s'%(name, voice)
    os.makedirs(dst_dir, exist_ok=True)
    json_filename = os.path.join(src_dir, '%s-%s-%s-extras.json'%(name, voice, target_emotion))
    with open(json_filename, 'r') as f:
        extras = json.load(f)

    best_audio_path = None
    best_score = float('-inf')
    for k in sorted(extras.keys()):
        if not int(extras[k]['qualified']):
            continue

        score = float(extras[k]['score'])
        if score > best_score:
            best_score = score
            best_audio_path = k

    if best_audio_path is not None:
        shutil.copy(os.path.join(src_dir, best_audio_path), os.path.join(dst_dir, best_audio_path))


def tts_copier():
    for name in ['long', 'noflights', 'hawaiiflight', 'hello', 'hi']:
        for voice in ['echo', 'alloy', 'ash']:
            for target_emotion in tqdm(['hesitant', 'frazzled', 'impatient', 'empathetic', 'bored', 'angry', 'neutral', 'sad', 'happy']):
                tts_copier_one(name, voice, target_emotion)


if __name__ == '__main__':
    tts_copier()
