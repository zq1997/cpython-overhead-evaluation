import json
import math
import subprocess

import numpy as np

import config
from util import saver


@saver.save_result_in_memory()
def list_meta():
    with open(config.CALIBRATE_JSON) as f:
        return json.load(f)


@saver.save_result_in_memory()
def list_names():
    return [x[0] for x in list_meta()]


def list_names_arr():
    return np.asarray((list_names()))


def run_bm(bm_meta, cpython=config.CPY_EXE, duration=config.RUN_DURATION,
           *, env=None, pre_args=(), stderr=True):
    _, single_duration, scr_args = bm_meta
    env = {
        'PYTHONPATH': config.PIP_DIR,
        **(env or {})
    }
    subprocess.check_call(
        [
            *pre_args,
            cpython, *scr_args,
            '--worker', '-n1', '-w0', '-l', str(math.ceil(duration / single_duration)),
            '--inherit-environ=' + ','.join(env.keys()),
        ],
        stdout=subprocess.DEVNULL,
        stderr=None if stderr else subprocess.DEVNULL,
        env=env
    )
