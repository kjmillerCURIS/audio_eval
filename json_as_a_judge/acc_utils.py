import os
import sys
import numpy as np
from scipy.stats import bootstrap


#returns a STRING
def compute_acc(credits, fancy=False):
    CI = bootstrap((credits,), np.mean).confidence_interval
#    print(0.5*(CI.low + CI.high))
#    print(np.mean(credits))
    if fancy: #\textbf{89.4} {\scriptsize (82.8–93.7)} (N{=}123)
        return '\\textbf{%.1f} {\\scriptsize (%.1f-%.1f)} (N{=}%d)'%(100.0 * np.mean(credits), 100.0 * CI.low, 100.0 * CI.high, len(credits))
    else:
        return '%.3f [%.3f, %.3f]'%(np.mean(credits), CI.low, CI.high)
