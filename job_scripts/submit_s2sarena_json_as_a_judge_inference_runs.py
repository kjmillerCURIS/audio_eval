import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_s2sarena_json_as_a_judge_inference.sh'
#PARAMS_KEY_LIST = ['v0_no_aux', 'v0_full_aux', 'v1a_no_aux', 'v1b_no_aux', 'v1c_no_aux', 'v1d_no_aux', 'v1e_no_aux', 'v1f_no_aux', 'v1g_no_aux']
PARAMS_KEY_LIST = ['v1c_no_aux']
REP_LIST = [1]


def submit_s2sarena_json_as_a_judge_inference_runs():
    for params_key in PARAMS_KEY_LIST:
        for rep in REP_LIST:
            job_name = 's2sarenajson_%s_rep%d'%(params_key, rep)
            my_cmd = 'qsub -N %s -v PARAMS_KEY=%s,REP=%d %s'%(job_name, params_key, rep, SCRIPT_NAME)
            print('submitting training run: "%s"'%(my_cmd))
            os.system(my_cmd)
            if DEBUG:
                print('DEBUG MODE: let\'s see how that first run goes...')
                return


if __name__ == '__main__':
    submit_s2sarena_json_as_a_judge_inference_runs()
