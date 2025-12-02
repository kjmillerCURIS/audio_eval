import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_run_alm_on_conversations.sh'
CHALLENGE_TYPE_LIST = ['pointwiseCR', 'pairwise', 'pointwiseRC']
PARA_CUE_TYPE_LIST = ['no_para_cue', 'soft_para_cue', 'hard_para_cue']
LAST_ONLY_LIST = [0, 1]
ALM_NAME_LIST = ['gemini-2.5-flash', 'gpt4o']


def submit_run_alm_on_conversations_runs():
    for alm_name in ALM_NAME_LIST:
        for challenge_type in CHALLENGE_TYPE_LIST:
            for para_cue_type in PARA_CUE_TYPE_LIST:
                for last_only in LAST_ONLY_LIST:
                    job_name = 'almconvo-%s-%s-last_only%d-%s'%(challenge_type, para_cue_type, last_only, alm_name)
                    my_cmd = 'qsub -N %s -v CHALLENGE_TYPE=%s,PARA_CUE_TYPE=%s,LAST_ONLY=%d,ALM_NAME=%s %s'%(job_name, challenge_type, para_cue_type, last_only, alm_name, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_run_alm_on_conversations_runs()
