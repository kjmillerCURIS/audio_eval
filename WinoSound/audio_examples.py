import os
import sys
import json
import shutil
from tqdm import tqdm
sys.path.append('.')
from WinoSound.tts_generator import TTSGenerator
from WinoSound.tts_scorer import TTSScorer
from WinoSound.compute_baseline_audio_stats import compute_stats_one_audio


CA0 = """
HUMAN(neutral): "I'm looking for a restaurant called The Missing Sock. Can you tell me its location?"
VA(neutral): "It's in the east, postcode CB25 9AQ, Finders Corner Newmarket Road. Anything else?"
HUMAN(hesitant): "Okay. Is that address in a crime-free area, and is it convenient to things to do?"
VA(empathetic): "I apologize, I don't have real-time crime data. However, I can search for reviews or news about the area's safety, and also look up nearby attractions for you. Would you like me to do that?"
"""
CB0 = """
HUMAN(neutral): "I'm looking for a restaurant called The Missing Sock. Can you tell me its location?"
VA(neutral): "It's in the east, postcode CB25 9AQ, Finders Corner Newmarket Road. Anything else?"
HUMAN(impatient): "Okay. Is that address in a crime-free area, and is it convenient to things to do?"
VA(neutral): "I cannot provide real-time crime statistics for locations. I can search for nearby attractions and points of interest if you wish."
"""
CA1 = """
HUMAN(happy): "It's great to see you, professor."
HUMAN(happy): "I want to talk to you about courses to select for the next term."
VA(happy): "Excellent! I'm eager to help you explore your options and plan an exciting term."
"""
CB1 = """
HUMAN(hesitant): "It's great to see you, professor."
HUMAN(hesitant): "I want to talk to you about courses to select for the next term."
VA(empathetic): "Of course. Please, take your time. We can discuss any concerns you have and find the best path forward."
"""
CA2 = """
HUMAN(hesitant):"what is the forecast in my city"
VA(neutral):"What city do you want the weather for?"
HUMAN(frazzled):"Inglewood, what's the weekly weather looking like?"
VA(empathetic):"I understand you're anxious about the forecast. There is a wide range of conditions next week, from 40-100F and snow to hot, but I can help you narrow it down. What details are most concerning to you?"
"""
CB2 = """
HUMAN(impatient):"what is the forecast in my city"
VA(neutral):"What city do you want the weather for?"
HUMAN(angry):"Inglewood, what's the weekly weather looking like?"
VA(neutral):"For Inglewood next week, conditions range from 40-100F with potential snow to hot weather. To give you the most relevant information, what specific details are you looking for?"
"""
VOICES = {'HUMAN' : 'ash', 'VA' : 'alloy'}


def process(conversation, conversation_name):
    attempts_dir = 'WinoSound/convo_attempts/%s'%(conversation_name)
    results_dir = 'WinoSound/convo_results/%s'%(conversation_name)
    os.makedirs(attempts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    my_generator = TTSGenerator()
    my_scorer = TTSScorer()
    turns = conversation.strip().split('\n')
    for t, turn in tqdm(enumerate(turns)):
        print(turn)
        ss = turn.split(':')
        assert(len(ss) == 2)
        text = ss[1].strip().strip('"')
        ss = ss[0].split('(')
        assert(len(ss) == 2)
        speaker = ss[0]
        voice = VOICES[speaker]
        target_emotion = ss[1].rstrip('):')
        name = '%s-turn%02d%s'%(conversation_name, t, speaker)
        best_audio_path = None
        best_score = float('-inf')
        if target_emotion in ['hesitant', 'frazzled', 'impatient', 'empathetic']: #need baselines
            print('generating baselines...')
            stats = []
            for rep in tqdm(range(10)):
                neutral_audio_path = '%s/%s-neutralrep%02d.wav'%(attempts_dir, name, rep)
                if os.path.exists(neutral_audio_path):
                    print('skipping generation (already exists)! "%s"'%(neutral_audio_path))
                    continue

                while True:
                    success = my_generator.generate(text, 'neutral', voice, neutral_audio_path)
                    if success:
                        break

                stats_one = compute_stats_one_audio(neutral_audio_path)
                stats.append(stats_one)

            stats_filename = '%s/%s-neutral-stats.json'%(attempts_dir, name)
            with open(stats_filename, 'w') as f:
                json.dump(stats, f, indent=4)

        extras = {}
        extras_filename = '%s/%s-extras.json'%(attempts_dir, name)
        if os.path.exists(extras_filename):
            with open(extras_filename, 'r') as f:
                extras = json.load(f)

        for rep in tqdm(range(10)):
            audio_path = '%s/%s-rep%02d.wav'%(attempts_dir, name, rep)
            if os.path.basename(audio_path) in extras:
                assert(os.path.exists(audio_path))
                print('skipping generation (already exists)! (and scoring too!) "%s"'%(audio_path))
                extra = extras[os.path.basename(audio_path)]
                qualified = extra['qualified']
                score = extra['score']
            else:
                while True:
                    success = my_generator.generate(text, target_emotion, voice, audio_path)
                    if success:
                        break

                qualified, score, extra = my_scorer.score(audio_path, target_emotion, None, name)

            print('%s: qualified=%d, score=%f'%(os.path.basename(audio_path), qualified, score))
            print(extra)
            extras[os.path.basename(audio_path)] = extra
            with open(extras_filename, 'w') as f:
                json.dump(extras, f, indent=4)

            if not qualified:
                continue

            if score > best_score:
                best_score = score
                best_audio_path = audio_path

        if best_audio_path is None:
            print('FAILED to get qualifier for "%s" "%s" "%s"'%(name, text, target_emotion))
        else:
            print('SUCCEEDED to get qualifier for "%s" "%s" "%s"'%(name, text, target_emotion))
            shutil.copy(best_audio_path, os.path.join(results_dir, os.path.basename(best_audio_path)))


def main():
#    process(CA0, 'CA0')
#    process(CB0, 'CB0')
#    process(CA1, 'CA1')
#    process(CB1, 'CB1')
    process(CA2, 'CA2')
#    process(CB2, 'CB2')


if __name__ == '__main__':
    main()
