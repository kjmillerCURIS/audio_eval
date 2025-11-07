import os
import sys
import json
import math
from tqdm import tqdm


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

#ambiguity_counter = 0 #keep track of how many examples aren't completely resolved by the constraints

#rating_vec should at least have key 'LLM_fused'
def llm_fusion_fn(rating_vec):
    return rating_vec['LLM_fused']

#same as "heuristic_tree()" rule that Venkatesh shared on slack on 9/29 around 3pm
#rating_vec should at least have keys 'content', 'voice_quality', and 'instruction_following_audio'
def venkatesh_heuristic_fn(rating_vec):
    if rating_vec['content'] == 'both_bad' and rating_vec['instruction_following_audio'] == 'both_bad':
        return 'both_bad'

    # Original Hierarchy
    for criteria in ['content', 'instruction_following_audio', 'voice_quality']:
        rating = rating_vec[criteria]
        if rating in ['1', '2']:
            return rating

    # Default Tie Handling
    if rating_vec['content'] == 'both_good':
        return 'both_good'

    return 'both_bad'

#not a rule function, just a helper
#will use "rules of thumb" to constrain the possible outputs, and return the feasible set as a list
def constraints_helper(rating_vec):
    #consensus rule
    if all([rating_vec[criteria] == rating_vec['content'] for criteria in ['content', 'instruction_following_audio', 'voice_quality']]):
        return [rating_vec['content']]

    #content-paralinguistic-pessimism rule
    content_points = RATING2POINTS[rating_vec['content']]
    instruction_points = RATING2POINTS[rating_vec['instruction_following_audio']]
    feasible_points = tuple([list(range(min(cp, ip) + 1)) for cp, ip in zip(content_points, instruction_points)])
    feasible_set = []
    for f1 in feasible_points[0]:
        for f2 in feasible_points[1]:
            feasible_set.append(POINTS2RATING[(f1, f2)])

    return feasible_set

#not a rull function, just a helper
def constraint_projection_helper(rating, feasible_set):
    best_dist = float('+inf')
    best_rating = None
    for x in feasible_set:
        pointsX = RATING2POINTS[x]
        pointsR = RATING2POINTS[rating]
        dist = math.fabs(pointsX[0] - pointsR[0]) + math.fabs(pointsX[1] - pointsR[1])
        if dist < best_dist:
            best_dist = dist
            best_rating = x

    return best_rating

#run venkatesh_heuristic_fn(), but then use some "rules of thumb" to constrain the possible outputs and pick the closest one to LLM fusion's output
#rating_vec should at least have keys 'content', 'voice_quality', 'instruction_following_audio', and 'LLM_fused'
def venkatesh_heuristic_with_constraints_fn(rating_vec):
    feasible_set = constraints_helper(rating_vec)
    assert(len(feasible_set) > 0)
    if len(feasible_set) == 1:
        return feasible_set[0]

#    global ambiguity_counter
#    ambiguity_counter += 1

    rating = venkatesh_heuristic_fn(rating_vec)
    if rating in feasible_set:
        return rating

    return constraint_projection_helper(rating, feasible_set)

#run llm_fusion_fn(), but then use some "rules of thumb" to constrain the possible outputs and pick the closest one to Venkatesh's output
#rating_vec should at least have keys 'content', 'voice_quality', and 'instruction_following_audio'
def llm_fusion_with_constraints_fn(rating_vec):
    feasible_set = constraints_helper(rating_vec)
    assert(len(feasible_set) > 0)
    if len(feasible_set) == 1:
        return feasible_set[0]

    rating = llm_fusion_fn(rating_vec)
    if rating in feasible_set:
        return rating

    return constraint_projection_helper(rating, feasible_set)


def measure_agreement_one_judge_one_rule(results_dict, annotation_dict, rule_fn, is_audiojudge):
#    if rule_fn == venkatesh_heuristic_with_constraints_fn:
#        global ambiguity_counter
#        ambiguity_counter = 0

    correct = 0
    correct_tie_aware = 0
    total = 0
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        output = results_dict['outputs'][k]
        if k not in annotation_dict:
            continue

        if not all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] != '' for a in CRITERIA]):
            assert(all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]))
            continue

        credit = 0
        credit_tie_aware = 0
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
            credit_tie_aware += 1.0 - math.fabs(TIE_AWARE_MAP[pred] - TIE_AWARE_MAP[gt])

        credit /= len(preds)
        credit_tie_aware /= len(preds)
        correct += credit
        correct_tie_aware += credit_tie_aware
        total += 1

#    if rule_fn == venkatesh_heuristic_with_constraints_fn:
#        if is_audiojudge:
#            print('%.1f cases where |feasible_set| > 1'%(ambiguity_counter / 2))
#        else:
#            print('%d cases where |feasible_set| > 1'%(ambiguity_counter))

    print(total)
    return correct / total, correct_tie_aware / total


def apply_rules_to_s2sarena():
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
    for rule_fn in [llm_fusion_fn, venkatesh_heuristic_fn, llm_fusion_with_constraints_fn, venkatesh_heuristic_with_constraints_fn]:
        AudioJudge_agreement = measure_agreement_one_judge_one_rule(AudioJudge_results_dict, annotation_dict, rule_fn, True)
        LLM_judge_agreement = measure_agreement_one_judge_one_rule(LLM_judge_results_dict, annotation_dict, rule_fn, False)
        JSON_judge_agreement = measure_agreement_one_judge_one_rule(JSON_judge_results_dict, annotation_dict, rule_fn, False)
        lines.append('"%s": AudioJudge = %.4f, LLM-judge = %.4f, JSON-judge = %.4f'%(rule_fn.__name__,AudioJudge_agreement[0], LLM_judge_agreement[0], JSON_judge_agreement[0]))
        lines_tie_aware.append('"%s": AudioJudge = %.4f, LLM-judge = %.4f, JSON-judge = %.4f'%(rule_fn.__name__,AudioJudge_agreement[1], LLM_judge_agreement[1], JSON_judge_agreement[1]))

    print('exact:')
    print('\n'.join(lines))
    print('tie-aware:')
    print('\n'.join(lines_tie_aware))


if __name__ == '__main__':
    apply_rules_to_s2sarena()
