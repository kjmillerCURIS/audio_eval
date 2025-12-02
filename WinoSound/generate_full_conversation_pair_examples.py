import os
import sys
import glob
import json
import random
import shutil
from string import Template
from tqdm import tqdm
sys.path.append('.')
from llm_utils import run_llm
from WinoSound.tts_generator import TTSGenerator, get_display_text
from WinoSound.tts_scorer import TTSScorer
from WinoSound.tts_baseline_utils import handle_baseline


TARGET_EMOTIONS = ['hesitant', 'frazzled', 'impatient', 'empathetic', 'neutral', 'angry', 'sad', 'happy']
EMPATHETIC_SYNONYMS = ['empathetic', 'apologetic', 'reassuring']
VOICES = ['echo', 'alloy', 'ash']
OVERRIDE_THRESHOLD = 3 #this many cropping failures before we just render without a template
EXPRESS_THRESHOLD = 3 #this many qualifications before we just return the best so far
NUM_ATTEMPTS = 10
DATA_STRIDE = 6


#return source_dataset, sampling_params
def load_source_dataset():
    with open('WinoSound/od3_datasets/od3_train.jsonl', 'r') as f:
        json_list = list(f)

    source_dataset = []
    for json_str in tqdm(json_list):
        source_dataset.append(json.loads(json_str))

    #'DSCT11-Track-5': 25579, 'kvret': 2425, 'MultiWOZ_2.2': 8425, 'NOESIS': 3967, 'SIMMC2.1': 7306
    sampling_params = [(0.4, ['DSCT11-Track-5', 'kvret', 'MultiWOZ_2.2', 'NOESIS', 'SIMMC2.1']), (0.3, ['SIMMC2.1', 'NOESIS', 'kvret']), (0.2, ['NOESIS', 'kvret']), (0.1, ['kvret'])]
    return source_dataset, sampling_params


def load_prompts():
    initial_prompt_filenames = ['WinoSound/prompts/inaug_prompt_emotion_only_v4_exclude_bored.txt', 'WinoSound/prompts/inaug_prompt_emotion_only_v4_exclude_bored_and_endimpatienthesitant.txt']
    refinement_prompt_filename = 'WinoSound/prompts/inaug_prompt_emotion_only_v4_refinement.txt'
    prompts = {'initial_prompts' : []}
    for fn in initial_prompt_filenames:
        with open(fn, 'r') as f:
            prompts['initial_prompts'].append(Template(f.read()))

    with open(refinement_prompt_filename, 'r') as f:
        prompts['refinement_prompt'] = f.read()

    return prompts


#returns source_conversation, index
#does NOT modify avoid_indices
def sample_source_conversation(source_dataset, sampling_params, avoid_indices, offset):
    r = random.uniform(0.0, 1.0)
    my_filter = None
    for prob, one_filter in sampling_params:
        if r <= prob:
            my_filter = one_filter
            break

        r -= prob

    while True:
        index = random.choice(range(len(source_dataset)))
        if index % DATA_STRIDE != offset:
            continue

        if index in avoid_indices:
            continue

        if source_dataset[index]['source_dataset'] not in my_filter:
            continue

        break

    lines = []
    for turn in source_dataset[index]['turns']:
        lines.append('%s: "%s"'%(['HUMAN', 'VA'][turn['is_agent']], turn['text']))

    source_conversation = '\n'.join(lines)
    return source_conversation, index


def parse_line(line):
    ss = line.split('(')
    speaker = ss[0]
    if speaker not in ['HUMAN', 'VA']:
        print('Unexpected speaker "%s"'%(speaker))
        return None

    line = '('.join(ss[1:])
    ss = line.split('):')
    target_emotion = ss[0]
    if target_emotion in TARGET_EMOTIONS:
        pass
    elif all([x in EMPATHETIC_SYNONYMS for x in target_emotion.split('/')]):
        target_emotion = 'empathetic'
    else:
        print('Unexpected target_emotion "%s"'%(target_emotion))
        return None

    line = '):'.join(ss[1:])
    text = line.strip().strip('"')
    display_text = get_display_text(text)
    return (speaker, target_emotion, text, display_text)


#returns what run_LLM_generation() will return, or None's if output cannot be parsed
def process_LLM_generator_output(output):
    lines = output.split('\n')
    conv_state = 'pre'
    A_lines = []
    B_lines = []
    for line in lines:
        if line.strip() == '':
            continue

        if '=======FINAL ANSWER=======' in line:
            if conv_state != 'pre':
                print('Unexpected LLM output - misplaced FINAL ANSWER')
                return None, None, None, None

            conv_state = 'A'
        elif line.strip() == '===':
            if conv_state != 'A':
                print('Unexpected LLM output - misplaced divider')
                return None, None, None, None

            conv_state = 'B'
        elif conv_state == 'A':
            A_lines.append(line.strip())
        elif conv_state == 'B':
            B_lines.append(line.strip())

    if len(A_lines) != len(B_lines):
        print('Unexpected LLM output - unequal lengths')
        return None, None, None, None

    if len(A_lines) < 2:
        print('Unexpected LLM output - too few turns')
        return None, None, None, None

    A_turns = []
    for line in A_lines:
        turn_tuple = parse_line(line)
        if turn_tuple is None:
            print('Unexpected LLM output - failed to parse line "%s"'%(line))
            return None, None, None, None

        A_turns.append(turn_tuple)

    B_turns = []
    for line in B_lines:
        turn_tuple = parse_line(line)
        if turn_tuple is None:
            print('Unexpected LLM output - failed to parse line "%s"'%(line))
            return None, None, None, None

        B_turns.append(turn_tuple)

    if not (A_turns[-2][0] == 'HUMAN' and A_turns[-1][0] == 'VA' and B_turns[-2][0] == 'HUMAN' and B_turns[-1][0] == 'VA'):
        print('Unexpected LLM output - last 2 turns not HUMAN -> VA')
        return None, None, None, None

    for turnA, turnB in zip(A_turns[:-1], B_turns[:-1]):
        if turnA[-1] != turnB[-1]:
            print(turnA[-1])
            print(turnB[-1])
            print('Unexpected LLM output - mismatching words in CA vs CB')
            return None, None, None, None

    return A_turns[:-1], A_turns[-1], B_turns[:-1], B_turns[-1]


#returns CA, RA, CB, RB
#CA, CB are lists of tuples of (speaker, target_emotion, text, display_text) where display_text has punctuation removed
#RA, RB are single tuples of the above
#target_emotion should be one of ['hesitant', 'frazzled', 'impatient', 'empathetic', 'neutral', 'angry', 'sad', 'happy']
#speaker should be one of ['HUMAN', 'VA']
#will run forever until we get a valid output (yeah, I know...)
def run_LLM_generation(source_conversation, prompts, log_filename):
    print('running LLM...')
    initial_prompt_choice = random.choice([0,1])
    initial_prompt = prompts['initial_prompts'][initial_prompt_choice]
    initial_prompt = initial_prompt.substitute(source_conversation=source_conversation)
    while True:
        responses = run_llm([initial_prompt, prompts['refinement_prompt']], llm_name='gemini-2.5-flash', skip_config=True)
        if any([r is None for r in responses]):
            print('ope, got a None response for some reason, let\'s try again!')
            continue

        CA, RA, CB, RB = process_LLM_generator_output(responses[1])
        if CA is None:
            print('LLM output had something unexpected, rerunning LLM...')
            continue

        print('CA:')
        print('\n'.join([str(x) for x in CA]))
        print('\nRA:')
        print(RA)
        print('\nCB:')
        print('\n'.join([str(x) for x in CB]))
        print('\nRB:')
        print(RB)
        print('')
        log_body = responses[0] + '\n\n=======AFTER REFINEMENT=======\n\n' + responses[1]
        preamble = '=======SOURCE CONVO=======\n\n' + source_conversation + '\n\nprompt_choice=%d\n\n'%(initial_prompt_choice) + '\n\n=======LLM INITIAL RESPONSE=======\n\n'
        with open(log_filename, 'w') as f:
            f.write(preamble + log_body)

        return CA, RA, CB, RB


#return output_filename or None if turn could not be rendered
def render_one_turn(turn_tuple, voice_dict, turn_name, attempt_dir, output_dir, offset):
    my_generator, my_scorer = TTSGenerator(), TTSScorer()
    speaker, target_emotion, text, _ = turn_tuple
    voice = voice_dict[speaker]
    handle_baseline(text, target_emotion, voice, turn_name, attempt_dir, offset)
    best_audio_path = None
    best_score = float('-inf')
    extras = {}
    extras_filename = os.path.join(attempt_dir, '%s-extras.json'%(turn_name))
    num_qualifications = 0
    for rep in tqdm(range(NUM_ATTEMPTS)):
        audio_path = os.path.join(attempt_dir, '%s-rep%02d.wav'%(turn_name, rep))
        overrides = 0
        while True:
            success = my_generator.generate(text, target_emotion, voice, audio_path, offset)
            if success:
                break

            overrides += 1
            if overrides >= OVERRIDE_THRESHOLD:
                print('enough attempts! just do without template!')
                success = my_generator.generate(text, target_emotion, voice, audio_path, offset, override_template=True)
                assert(success)
                break

        qualified, score, extra = my_scorer.score(audio_path, target_emotion, None, turn_name)
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

        num_qualifications += 1
        if num_qualifications >= EXPRESS_THRESHOLD:
            print('enough qualifications! just use the best so far!')
            break

    if best_audio_path is None:
        print('FAILED to get qualifier for "%s" "%s" "%s"'%(turn_name, text, target_emotion))
        output_audio_path = None
    else:
        print('SUCCEEDED to get qualifier for "%s" "%s" "%s"'%(turn_name, text, target_emotion))
        output_audio_path = os.path.join(output_dir, os.path.basename(best_audio_path))
        shutil.copy(best_audio_path, output_audio_path)

    return output_audio_path


#return success (if False then caller should quarantine output_dir)
#this should setup the json that website would need, which might contain other info too
def render_conversation_pair(CA, RA, CB, RB, attempt_dir, index, output_dir, offset):
    success = True
    info = {}
    info['response_display_order'] = random.sample(['RA', 'RB'], 2)
    [human_voice, va_voice] = random.sample(VOICES, 2)
    voice_dict = {'HUMAN' : human_voice, 'VA' : va_voice}
    info['voice_dict'] = voice_dict
    for AB, CACB, RARB in zip(['A', 'B'], [CA, CB], [RA, RB]):
        info['C' + AB] = []
        for t, turn_tuple in tqdm(enumerate(CACB)):
            speaker, target_emotion, text, display_text = turn_tuple
            turn_info = {'speaker' : speaker, 'target_emotion' : target_emotion, 'text' : text, 'display_text' : display_text}
            turn_name = '%d-C%s-turn%02d%s'%(index, AB, t, speaker)
            audio_path = render_one_turn(turn_tuple, voice_dict, turn_name, attempt_dir, output_dir, offset)
            if audio_path is not None:
                audio_path = os.path.relpath(audio_path, start=os.path.join(output_dir, '..'))
                turn_info['audio_path'] = audio_path
                info['C' + AB].append(turn_info)
            else:
                turn_info['audio_path'] = 'NA'
                info['C' + AB].append(turn_info)
                with open(os.path.join(output_dir, '%d-info.json'%(index)), 'w') as f:
                    json.dump(info, f)

                return False

        speaker, target_emotion, text, display_text = RARB
        turn_info = {'speaker' : speaker, 'target_emotion' : target_emotion, 'text' : text, 'display_text' : display_text}
        turn_name = '%d-R%s-turn%02d%s'%(index, AB, t, speaker)
        audio_path = render_one_turn(RARB, voice_dict, turn_name, attempt_dir, output_dir, offset)
        if audio_path is not None:
            audio_path = os.path.relpath(audio_path, start=os.path.join(output_dir, '..'))
            turn_info['audio_path'] = audio_path
            info['R' + AB] = turn_info
        else:
            turn_info['audio_path'] = 'NA'
            info['R' + AB] = turn_info
            with open(os.path.join(output_dir, '%d-info.json'%(index)), 'w') as f:
                json.dump(info, f)

            return False

    with open(os.path.join(output_dir, '%d-info.json'%(index)), 'w') as f:
        json.dump(info, f, indent=4)

    return True


def generate_full_conversation_pair_examples(num_examples, random_seed, output_parent_dir, offset):
    num_examples = int(num_examples)
    random_seed = int(random_seed)
    offset = int(offset)

    random.seed(random_seed)
    os.makedirs(output_parent_dir, exist_ok=True)
    website_dir = os.path.join(output_parent_dir, os.path.basename(output_parent_dir) + '-for_website')
    os.makedirs(website_dir, exist_ok=True)
    attempt_dir = os.path.join(output_parent_dir, os.path.basename(output_parent_dir) + '-tts_attempts')
    os.makedirs(attempt_dir, exist_ok=True)
    failure_dir = os.path.join(output_parent_dir, os.path.basename(output_parent_dir) + '-failures')
    os.makedirs(failure_dir, exist_ok=True)

    source_dataset, sampling_params = load_source_dataset()
    prompts = load_prompts()
    harvest_indices = lambda my_dir : set([int(os.path.basename(x)) for x in sorted(glob.glob(os.path.join(my_dir, '*'))) if os.path.basename(x).isdigit()])
    avoid_indices = harvest_indices(website_dir) | harvest_indices(failure_dir)
    print('AVOID INDICES: %s'%(str(avoid_indices)))
    for _ in tqdm(range(num_examples)):
        while True: #break on success
            source_conversation, index = sample_source_conversation(source_dataset, sampling_params, avoid_indices, offset)
            avoid_indices.add(index) #yes, this means we might avoid certain kinds of conversations
            example_dir = os.path.join(website_dir, str(index))
            os.makedirs(example_dir, exist_ok=True)
            print('running LLM...')
            LLM_log_filename = os.path.join(example_dir, '%d-LLM_log.txt'%(index))
            CA, RA, CB, RB = run_LLM_generation(source_conversation, prompts, LLM_log_filename)
            success = render_conversation_pair(CA, RA, CB, RB, attempt_dir, index, example_dir, offset)
            if not success:
                print('failed to render %d'%(index))
                dst_dir = os.path.join(failure_dir, str(index))
                assert(not os.path.exists(dst_dir))
                shutil.move(example_dir, dst_dir)
            else:
                break


def usage():
    print('Usage: python WinoSound/generate_full_conversation_pair_examples.py <num_examples> <random_seed> <output_parent_dir> <offset>')


if __name__ == '__main__':
    generate_full_conversation_pair_examples(*(sys.argv[1:]))
