import os
import sys
import json
from tqdm import tqdm


def is_failure(output):
    pred_label = output['pred']['label']
    assert(pred_label in ['1', '2'])
    pred_label = int(pred_label)
    gt = output['example_info']['chosen_model']
    assert(gt in ['A', 'B'])
    return (['A', 'B'][pred_label - 1] != gt)


def get_s2sarena_json_failures(results_dict_filename, failure_dir):
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    print(len(results_dict['outputs']))

    #sort into buckets
    buckets = {}
    for k in tqdm(sorted(results_dict['outputs'], key=int)):
        output = results_dict['outputs'][k]
        if 'language' not in output['example_info']:
            language = 'NA'
            print('!')
        else:
            language = output['example_info']['language']

        bucket_id = (output['example_info']['task'], language)
        if bucket_id not in buckets:
            buckets[bucket_id] = {'successes' : {}, 'failures' : {}}

        if is_failure(output):
            buckets[bucket_id]['failures'][k] = output
        else:
            buckets[bucket_id]['successes'][k] = output


    #print stats
    for bucket_id in sorted(buckets.keys(), key=lambda x: len(buckets[x]['failures']), reverse=True):
        num_successes = len(buckets[bucket_id]['successes'])
        num_failures = len(buckets[bucket_id]['failures'])
        num_total = num_successes + num_failures
        print(bucket_id)
        print('%d / %d failures (%.1f %%)'%(num_failures, num_total, 100.0 * num_failures / num_total))
        print('')


    #arrange files
    for bucket_id in sorted(buckets.keys()):
        s = bucket_id[0].replace(' ', '_') + '-' + bucket_id[1]
        for successes_or_failures in ['successes', 'failures']:
            dst_dir = os.path.join(failure_dir, s, successes_or_failures)
            os.makedirs(dst_dir, exist_ok=True)
            for k in sorted(buckets[bucket_id][successes_or_failures].keys()):
                output = buckets[bucket_id][successes_or_failures][k]
                dst_filename = os.path.join(dst_dir, '%s-%s-%s.json'%(s, successes_or_failures[:-2], k))
                with open(dst_filename, 'w') as f:
                    json.dump(output, f, indent=2)


def usage():
    print('Usage: python get_s2sarena_json_failures.py <results_dict_filename> <failure_dir>')


if __name__ == '__main__':
    get_s2sarena_json_failures(*(sys.argv[1:]))
