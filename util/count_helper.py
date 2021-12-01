import os
import multiprocessing

import numpy as np

from util import benchmarks
from util import saver
from util import log
import config


def _do_run(args):
    cpython, output_file, bm_meta = args

    if not os.path.exists(output_file):
        logger = log.TimeLogger(bm_meta[0])
        with logger.add_message('warm'):
            benchmarks.run_bm(bm_meta, cpython, config.WARM_UP_DURATION)
        with logger.add_message('output'):
            with saver.TildeFile(output_file) as output_file_:
                for _ in range(config.COUNT_REPEAT):
                    benchmarks.run_bm(bm_meta, cpython, env={'OUTPUT_FILE': output_file_})


def collect(data_name):
    cpython = config.cpy_exe_path(data_name)
    file_prefix = saver.ensure_dir(config.data_path(data_name, ''), check=False)
    args_list = []
    for bm_meta in benchmarks.list_meta():
        args_list.append([cpython, file_prefix + bm_meta[0], bm_meta])

    with multiprocessing.Pool(config.PARALLEL) as pool:
        log.title('instrumentation count: ' + data_name)
        pool.map(_do_run, args_list)


@saver.save_result_as_pickle('', lambda d: d)
def load(data_name):
    data = []
    bm_names = benchmarks.list_names()
    for bm_name in bm_names:
        with open(config.data_path(data_name, bm_name), 'rb') as f:
            data.append(np.frombuffer(f.read(), dtype=np.int_))
    return np.asarray(data)
