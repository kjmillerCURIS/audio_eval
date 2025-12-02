import os
import sys
import copy
import math
from tqdm import tqdm
from pydub import AudioSegment
import shutil
import spacy
import torchaudio
sys.path.append('.')
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
from WinoSound.asr_v2 import whisper_transcribe
from WinoSound.syllable_counter import count_total_syllables


TEMP_PATH = 'WinoSound/temp%02d.wav'


nlp = spacy.blank('en')


#FIXME: go back and pare down these templates. Some you could throw out entirely, others maybe a small thing at the beginning for the "burn-in" effect. See "tts_samples_notemplate" for what the "long" text sounds like without any template ("hello" and "hi" were already templateless). Of course, you'll have to be careful when it comes time to add emphasis.
INPUT_TEMPLATES = {
        'hesitant' : 'I am feeling hesitant and am speaking haltingly, you can hear how unsure I am when I say the following: "%s?" I\'m sorry but I just can\'t say it confidently.',
        'neutral' : 'Note how I speak in a neutral, professional, and clear tone when I say the following: "%s" Note how professional I sound.',
        'bored' : 'I\'m bored, this is so boring, you can hear how bored and low-energy I am when I say the following: "%s" oh man I could die of boredom',
        'frazzled' : 'Oh shit! I\'m so frazzled, I\'m panicking, it feels like everything is on fire! You can hear how panicked I am when I say the following: "%s" we are so fucking screwed!',
        'impatient' : 'I really don\'t have time for this, I\'m trying to keep my cool but I\'m in a hurry which is why I talk very fast when I say the following: "%s" can you please just cut to the chase?',
        'empathetic' : 'I\'m so sorry to hear that, my child. I totally understand why you\'re upset, and I hope my warm, apologetic tone brings you some comfort when I say the following: "%s" Now let\'s read a bedtime story, I promise things will get better.',
        'sad' : 'I\'m so sad I\'m crying, you can hear me sobbing through the following: "%s" Sorry, I need a tissue!',
        'happy' : 'I\'m so happy I could jump for joy! You can hear my bright, cheerful tone of voice when I say: "%s" Yay! Life is good!',
        'angry' : 'I\'m so pissed off! I\'m furious! You can hear how fucking angry I am when I yell: "%s!" I could fucking punch this stupid robot!',
    }
INSTRUCTIONS = {
        'hesitant' : 'Speak with a hesitant, halting tone of voice, as if unsure of what you are saying. Take plenty of pauses, and end EVERY sentence with an upward intonation, like it\'s a question.',
        'neutral' : 'Speak in a neutral, professional, and clear tone of voice.',
        'bored' : 'Speak in a very bored tone of voice, with very low energy, kind of slow.',
        'frazzled' : 'Speak in a very frazzled, panicked tone of voice, as if everything is on fire. Talk very very quickly, rushing through your words, and in a raised pitch.',
        'impatient' : 'Speak in an impatient tone of voice, like you\'re in a big hurry but trying to keep your cool. Talk very very quickly, rushing through your words, but keep your pitch normal.',
        'empathetic' : 'Speak in a very warm, apologetic, reassuring tone, like a parent comforting their upset child. Make sure the comforting tone really comes through.',
        'sad' : 'Speak in a very sad, sobbing tone of voice, with very audible sobbing. Exaggerate the sadness and crying.',
        'happy' : 'Speak in a very happy, cheerful, exuberant, overjoyed tone of voice, with very audible laughter. Exaggerate the cheerfulness and laughter.',
        'angry' : 'Yell in an extremely angry, furious, violent tone of voice, exaggerating your rage. Make sure it really sounds angry, like you\'re gonna fucking kill someone!',
    }
SPECIAL_INSTRUCTIONS = {
        ('ash', 'happy') : 'Speak in a calm, relaxed, happy, cheerful, joyous tone of voice, with audible laughter.', #ash in particular sometimes sounds borderline angry when given the common 'happy' instructions
    }
CROP_PREFIXES = {
        'hesitant' : 'say the following',
        'neutral' : 'say the following',
        'bored' : 'say the following',
        'frazzled' : 'say the following',
        'impatient' : 'say the following',
        'empathetic' : 'say the following',
        'sad' : 'through the following',
        'happy' : 'when I say',
        'angry' : 'when I yell',
    }
CROP_SUFFIXES = {
        'hesitant' : 'I\'m sorry but',
        'neutral' : 'Note how professional',
        'bored' : 'oh man I',
        'frazzled' : 'we are so',
        'impatient' : 'can you please',
        'empathetic' : 'Now let\'s read',
        'sad' : 'Sorry, I need',
        'happy' : 'Yay! Life is',
        'angry' : 'I could fucking',
    }
CROP_TOLERANCE = 2
SHORT_MODE_WORD_THRESHOLD = 2 #<= this many words means do short mode template for better cropping
SHORT_MODE_SYLLABLE_THRESHOLD = 3 #<= this many total syllables means do short mode template for better cropping
NEVER_USE_TEMPLATE = False


client = OpenAI()


#returns index such that you could use them to isolate out the payload
#for the last, it'll be a negative index
def match_cropping_target(text, target, first_or_last):
    assert(first_or_last in ['first', 'last'])
    def split_and_sanitize(s):
        ss = s.split()
        return [x.lower().strip('"\',.!?/\\()-:;$') for x in ss]

    text_list = split_and_sanitize(text)
    target_list = split_and_sanitize(target)
    if first_or_last == 'first':
        indices_to_consider = range(0, len(text_list) - len(target_list) + 1)
    elif first_or_last == 'last':
        indices_to_consider = range(-1 - len(target_list) + 1, -len(text_list) - 1, -1)
    else:
        assert(False)

    for i in indices_to_consider:
        is_match = True
        for j in range(len(target_list)):
            if text_list[i + j] != target_list[j]:
                is_match = False
                break

        if is_match:
            if first_or_last == 'first':
                return i + len(target_list)
            elif first_or_last == 'last':
                return i
            else:
                assert(False)

    return None


def run_tts(text, target_emotion, voice, audio_path, override_template=False):
    my_template = INPUT_TEMPLATES[target_emotion]
    if NEVER_USE_TEMPLATE or override_template or (len(text.split()) <= SHORT_MODE_WORD_THRESHOLD or count_total_syllables(text) <= SHORT_MODE_SYLLABLE_THRESHOLD):
        should_crop = False
        my_input = text
    else:
        should_crop = True
        my_input = my_template % text

    if (voice, target_emotion) in SPECIAL_INSTRUCTIONS:
        my_instructions = SPECIAL_INSTRUCTIONS[(voice, target_emotion)]
    else:
        my_instructions = INSTRUCTIONS[target_emotion]

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=my_input,
        instructions=my_instructions,
    ) as response:
        response.stream_to_file(audio_path)

    return should_crop


#removes any punctuation that might give away paralinguistics (although LLM usually keeps punctuation exactly the same)
def get_display_text(text):
    doc = nlp(text)
    prewords = [t.text.lower() for t in doc if not t.is_punct]
    words = []
    i = 0
    while i < len(prewords):
        if i < len(prewords) - 1 and prewords[i+1] in ['n\'t', '\'re', '\'ve', '\'s', '\'d', '\'m', '\'ll']:
            words.append(prewords[i] + prewords[i+1])
            i += 2
        else:
            words.append(prewords[i])
            i += 1

    return ' '.join(words)


def crop_audio(input_path, text, target_emotion, output_path):
    my_input = INPUT_TEMPLATES[target_emotion] % text
    expected_start = match_cropping_target(my_input, CROP_PREFIXES[target_emotion], 'first')
    expected_end = match_cropping_target(my_input, CROP_SUFFIXES[target_emotion], 'last')
    assert(expected_start is not None and expected_end is not None)
    transcript, word_chunks = whisper_transcribe(input_path)
    print(' '.join([wc[0] for wc in word_chunks]))
    start = match_cropping_target(' '.join([wc[0] for wc in word_chunks]), CROP_PREFIXES[target_emotion], 'first')
    end = match_cropping_target(' '.join([wc[0] for wc in word_chunks]), CROP_SUFFIXES[target_emotion], 'last')
    if start is None or end is None:
        print('missing start or end target in cropping')
        if get_display_text(transcript) == get_display_text(text):
            print('allow it anyway because the TTS basically ignored the template')
            shutil.copy(input_path, output_path)
            return True

        return False

    if math.fabs(start - expected_start) > CROP_TOLERANCE or math.fabs(end - expected_end) > CROP_TOLERANCE:
        print('unexpected location for start or end target in cropping')
        return False

    start_ms = 1000 * 0.5 * (word_chunks[start][1] + word_chunks[start - 1][2])
    end_ms = 1000 * 0.5 * (word_chunks[end][1] + word_chunks[end - 1][2])
    audio = AudioSegment.from_file(input_path)
    audio = audio[start_ms:end_ms]
    audio.export(output_path, format=os.path.splitext(output_path)[-1].replace('.', ''))
    return True


class TTSGenerator:
    def __init__(self):
        pass

    def generate(self, text, target_emotion, voice, audio_path, offset, override_template=False):
        assert(not os.path.exists(TEMP_PATH % offset))
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        should_crop = run_tts(text, target_emotion, voice, TEMP_PATH % offset, override_template=override_template)
        if should_crop:
            success = crop_audio(TEMP_PATH % offset, text, target_emotion, audio_path)
        else:
            success = True
            shutil.copy(TEMP_PATH % offset, audio_path)

        os.remove(TEMP_PATH % offset)
        return success


def main():
    my_generator = TTSGenerator()
    for name, text in [('furniturecope', 'Ok. Well I really like the bed to the left of that chair, can you put it in my cart? Also that coffee table beside the other bed is nice, might as well get that one too!'), ('hawaiiflight', 'I am flying to Hawaii tomorrow'), ('noflights', 'There are no flights on that day. Are there any other days that would work for you?'), ('long', 'The quick brown fox jumps over the lazy dog because the precipitation in Spain stays mainly in the plain'), ('hello', 'hello'), ('hi', 'hi')]:
        if NEVER_USE_TEMPLATE and name != 'long':
            continue

        for voice in ['echo', 'alloy', 'ash']:
            for target_emotion in ['sad']: #['hesitant', 'neutral', 'bored', 'frazzled', 'impatient', 'empathetic', 'sad', 'happy', 'angry']:
                print((name, voice, target_emotion))
                for rep in tqdm(range(10)):
                    out_dir = 'tts_samples'
                    if NEVER_USE_TEMPLATE:
                        out_dir = 'tts_samples_notemplate'

                    audio_path = 'WinoSound/%s/%s-%s-%s-%02d.wav'%(out_dir, name, voice, target_emotion, rep)
                    if os.path.exists(audio_path):
                        print('skipping! (already exists)')
                        continue

                    while True:
                        success = my_generator.generate(text, target_emotion, voice, audio_path)
                        if success:
                            break


if __name__ == '__main__':
    main()
