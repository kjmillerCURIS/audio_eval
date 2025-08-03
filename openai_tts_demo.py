import os
import sys
from pydub import AudioSegment
import torch
import torchaudio
import whisperx
from tqdm import tqdm
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


client = OpenAI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)


TEMP_FILENAME = 'temp.wav'


def extract_emphasis_words(text_with_emphasis):
    emphasis_words = [w.replace('*', '').strip('?!.,:;"\'()') for w in text_with_emphasis.split() if '*' in w]
    return emphasis_words


def adverbify(emotion):
    if emotion[-1] == 'y':
        return emotion[:-1] + 'ily'
    else:
        return emotion + 'ly'


def get_input_and_instructions(text_with_emphasis, emotion):
    emphasis_words = extract_emphasis_words(text_with_emphasis)
    emotion_adv = adverbify(emotion)
    if len(emphasis_words) == 0:
        my_input = f'Ron said {emotion_adv}, "I feel {emotion}. {text_with_emphasis}"'
        my_instructions = f'Narrate this sentence from an audiobook, making sure Ron sounds {emotion}. Exaggerate Ron\'s emotions.'
        return my_input, my_instructions

    if len(emphasis_words) == 1:
        emphasis_instruction = f' "{emphasis_words[0]}"'
    elif len(emphasis_words) == 2:
        emphasis_instruction = f's "{emphasis_words[0]}" and "{emphasis_words[1]}"'
    else:
        list_part = ', '.join([f'"{w}"' for w in emphasis_words[:-1]])
        emphasis_instruction = f's {list_part}, and "{emphasis_words[-1]}"'

    my_input = f'Heavily emphasizing the word{emphasis_instruction} in particular, Ron said {emotion_adv}, "I feel {emotion}. {text_with_emphasis}"'
    my_instructions = f'Narrate this sentence from an audiobook, making sure Ron sounds {emotion} and emphasizes the word{emphasis_instruction}, and no other words. Exaggerate Ron\'s emotions and emphasis.'
    return my_input, my_instructions


#keep the last N words
def crop_audio(transcript, N, output_filename):
    waveform, sample_rate = torchaudio.load(TEMP_FILENAME)
    duration = waveform.shape[1] / sample_rate
    custom_segment = {
        "start": 0.0,
        "end": duration,
        "text": transcript
    }

    alignment = whisperx.align([custom_segment], model_a, metadata, TEMP_FILENAME, device)
    assert(len(alignment['word_segments']) == len(transcript.split()))
    audio = AudioSegment.from_file(TEMP_FILENAME)
    crop_start_ms = 500 * (alignment['word_segments'][-N]['start'] + alignment['word_segments'][-(N+1)]['end'])
    audio = audio[crop_start_ms:]
    audio.export(output_filename, format=os.path.splitext(output_filename)[-1].replace('.', ''))


def tts_prompt_hack(text_with_emphasis, emotion, output_filename):
    my_input, my_instructions = get_input_and_instructions(text_with_emphasis, emotion)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=my_input,
        instructions=my_instructions,
    ) as response:
        response.stream_to_file(TEMP_FILENAME)

    crop_audio(my_input, len(text_with_emphasis.split()), output_filename)


if __name__ == '__main__':
    def make_output_filename(description, text_with_emphasis, emotion, rep):
        emphasis_words = extract_emphasis_words(text_with_emphasis)
        emphasis_part = '_'.join([w for w in emphasis_words])
        return os.path.join('tts_prompt_hack_outputs', f'{description}-emphasize_{emphasis_part}-emotion_{emotion}-rep{rep}.wav')

    descriptions_and_texts = []
    descriptions_and_texts.append(('blood_example', '**YOU** want my blood?'))
    descriptions_and_texts.append(('blood_example', 'You **WANT** my blood?'))
    descriptions_and_texts.append(('blood_example', 'You want **MY** blood?'))
    descriptions_and_texts.append(('blood_example', 'You want my **BLOOD**?'))
    descriptions_and_texts.append(('blood_example', '**YOU** want my **BLOOD**?'))
    descriptions_and_texts.append(('blood_example', '**YOU** want **MY** blood?'))
    descriptions_and_texts.append(('appointment_example', 'It **SAYS** my appointment is cancelled now.'))
    descriptions_and_texts.append(('appointment_example', 'It **SAYS** my appointment is **CANCELLED** now.'))
    descriptions_and_texts.append(('appointment_example', 'It **SAYS** my **APPOINTMENT** is **CANCELLED** now.'))
    descriptions_and_texts.append(('appointment_example', 'It says **MY** appointment **IS** cancelled **NOW**.'))
    for description, text_with_emphasis in tqdm(descriptions_and_texts):
        for emotion in ['confused', 'happy', 'angry', 'sad', 'neutral']:
            for rep in range(3):
                output_filename = make_output_filename(description, text_with_emphasis, emotion, rep)
                tts_prompt_hack(text_with_emphasis, emotion, output_filename)
