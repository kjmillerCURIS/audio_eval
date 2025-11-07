import os
import sys
import json
import math
from tqdm import tqdm
from acc_utils import compute_acc


CRITERIA = ['content', 'voice_quality', 'instruction_following_audio', 'overall (looking at dimensions)']
CRITERIA_MAP = {'content' : 'content', 'voice_quality' : 'voice quality', 'instruction_following_audio' : 'paralinguistic EQ/IF', 'overall (looking at dimensions)' : 'overall (looking at dimensions)'}
RATING_MAP = {'1' : '1', '2' : '2', '+' : 'both_good', '-' : 'both_bad'}
TIE_AWARE_MAP = {'1' : 0.0, '2' : 1.0, 'both_good' : 0.5, 'both_bad' : 0.5, 'tie' : 0.5}


'''
Rule functions
each one takes in a dict mapping {'content', 'voice_quality', 'instruction_following_audio', 'LLM_fused'} (or some subset thereof) to ratings
each rating being in {'1', '2', 'both_good', 'both_bad'}
return one of {'1', '2', 'both_good', 'both_bad'}
'''

RATING2POINTS = {'1' : (1, 0), '2' : (0, 1), 'both_good' : (1, 1), 'both_bad' : (0, 0)}
POINTS2RATING = {(1, 0) : '1', (0, 1) : '2', (1, 1) : 'both_good', (0, 0) : 'both_bad'}

def rating_min(ratingA, ratingB):
    pA, pB = RATING2POINTS[ratingA], RATING2POINTS[ratingB]
    pC = (min(pA[0], pB[0]), min(pA[1], pB[1]))
    return POINTS2RATING[pC]

def heuristic_tree_s2s_fn(rating_vec):
    #cap the final result to no better than content or EQ
    acceptability_cap = rating_min(rating_vec['content'], rating_vec['instruction_following_audio'])

    #do a cascade through the dimensions to see if any of them decide
    for criteria in ['content', 'instruction_following_audio', 'voice_quality']:
        rating = rating_vec[criteria]
        if rating in ['1', '2']:
            return rating_min(rating, acceptability_cap) #apply cap

    #if all dimensions are ties, go with capped content
    return rating_min(rating_vec['content'], acceptability_cap)

def majority_helper(votes):
    assert(len(votes) == 3)
    votes = [RATING2POINTS[vote] for vote in votes]
    avg = tuple([int(round(sum([vote[i] for vote in votes]) / len(votes))) for i in range(2)])
    return POINTS2RATING[avg]

#this is just to verify that the average-and-round method lines up with intuition
def majority_helper_test():
    assert(majority_helper(['1', '1', 'both_good']) == '1') #duplicate wins
    assert(majority_helper(['2', 'both_bad', 'both_bad']) == 'both_bad') #duplicate wins
    assert(majority_helper(['2', 'both_good', '1']) == 'both_good') #if both numbers present, then the tie that's present wins
    assert(majority_helper(['2', 'both_good', 'both_bad']) == '2') #if both ties present, then the number that's present wins

def majority_vote_fn(rating_vec):
    majority_helper_test()
    votes = [rating_vec[a] for a in ['content', 'instruction_following_audio', 'voice_quality']]
    return majority_helper(votes)


def measure_agreement_one_judge_one_rule(results_dict, annotation_dict, rule_fn, is_audiojudge):
    credits = []
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        output = results_dict['outputs'][k]
        if k not in annotation_dict:
            continue

        if not all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] != '' for a in CRITERIA]):
            assert(all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]))
            continue

        credit = 0
        preds = []
        if is_audiojudge:
            for swap in ['12', '21']:
                rating_vec = {criteria : output['response' + swap]['response'][criteria] for criteria in CRITERIA if criteria != 'overall (looking at dimensions)'}
                rating_vec['LLM_fused'] = output['fusion_part']['response' + swap]['pred']['overall_label']
                pred = rule_fn(rating_vec)
                assert(pred in ['1', '2', 'both_good', 'both_bad'])
                if swap == '21' and pred in ['1', '2']:
                    pred = {'1' : '2', '2' : '1'}[pred]

                preds.append(pred)
        else:
            rating_vec = {criteria : output['pred'][criteria] for criteria in CRITERIA if criteria != 'overall (looking at dimensions)'}
            rating_vec['LLM_fused'] = output['fusion_part']['pred']['overall_label']
            pred = rule_fn(rating_vec)
            assert(pred in ['1', '2', 'both_good', 'both_bad'])
            preds.append(pred)

        gt = annotation_dict[k]['ratings']['overall (looking at dimensions)']
        gt = RATING_MAP[gt]
        for pred in preds:
            credit += int(pred == gt)

        credit /= len(preds)
        credits.append(credit)

    print(len(credits))
    return compute_acc(credits)


def apply_rules_to_s2sarena_final():
    data_dir = 'json_as_a_judge/s2sarena_experiments'
    annotation_dict_filename = os.path.join(data_dir, 's2sarena_annotations_kevin.json')
    AudioJudge_results_dict_filename = os.path.join(data_dir, 'audiojudge_results', 'audiojudge_results_gemini-2.5-flash_icl0_dimensionwise1-with_fusion.json')
    LLM_judge_results_dict_filename = os.path.join(data_dir, 'results', 'results-v2_LLM_judge-gemini-2.5-flash-dimensionwise1-rep0-with_fusion.json')
    JSON_judge_results_dict_filename = os.path.join(data_dir, 'results', 'results-v2-gemini-2.5-flash-dimensionwise1-rep0-with_fusion.json')
    with open(annotation_dict_filename, 'r') as f:
        annotation_dict = json.load(f)

    with open(AudioJudge_results_dict_filename, 'r') as f:
        AudioJudge_results_dict = json.load(f)

    with open(LLM_judge_results_dict_filename, 'r') as f:
        LLM_judge_results_dict = json.load(f)

    with open(JSON_judge_results_dict_filename, 'r') as f:
        JSON_judge_results_dict = json.load(f)

    lines = []
    lines_tie_aware = []
    for rule_fn in [heuristic_tree_s2s_fn, majority_vote_fn]:
        AudioJudge_agreement = measure_agreement_one_judge_one_rule(AudioJudge_results_dict, annotation_dict, rule_fn, True)
        LLM_judge_agreement = measure_agreement_one_judge_one_rule(LLM_judge_results_dict, annotation_dict, rule_fn, False)
        JSON_judge_agreement = measure_agreement_one_judge_one_rule(JSON_judge_results_dict, annotation_dict, rule_fn, False)
        lines.append('"%s": AudioJudge = %s, LLM-judge = %s, JSON-judge = %s'%(rule_fn.__name__, AudioJudge_agreement, LLM_judge_agreement, JSON_judge_agreement))

    print('\n'.join(lines))


if __name__ == '__main__':
    apply_rules_to_s2sarena_final()
