import os
import multiprocessing

import numpy as np

from util import benchmarks
from util import saver
from util import log
import config


def _do_stat(args):
    file_prefix, cpython, event, bm_meta = args
    common_perf_args = ('perf', 'stat', '-x,', '-e', event)
    csv_file = file_prefix + '.csv'
    if not os.path.exists(csv_file):
        logger = log.TimeLogger(bm_meta[0])
        with logger.add_message('warm'):
            perf_args = common_perf_args + ('--',)
            for _ in range(2):
                benchmarks.run_bm(bm_meta, cpython, config.WARM_UP_DURATION, pre_args=perf_args, stderr=False)
        with logger.add_message('stat'):
            with saver.TildeFile(csv_file) as csv_file_:
                for i in range(config.STAT_REPEAT):
                    perf_args = common_perf_args + ('-o', csv_file_, '--append', '--')
                    benchmarks.run_bm(bm_meta, cpython, pre_args=perf_args)


def collect(data_name, event, cpython=config.CPY_EXE):
    file_prefix = saver.ensure_dir(config.data_path(data_name, ''), check=False)
    record_args_list = []
    for bm_meta in benchmarks.list_meta():
        bm_file_prefix = file_prefix + bm_meta[0]
        record_args_list.append([bm_file_prefix, cpython, event, bm_meta])

    with multiprocessing.Pool(config.PARALLEL) as pool:
        log.title('perf stat: ' + data_name)
        pool.map(_do_stat, record_args_list)


@saver.save_result_as_pickle('', lambda d: d)
def load(data_name):
    data = []
    for bm_name in benchmarks.list_names():
        data.append([])
        with open(config.data_path(data_name, bm_name + '.csv'), 'rt') as f:
            for line in f:
                if not line.isspace() and not line.startswith('#'):
                    data[-1].append(int(line.split(',', 1)[0]))
    return np.asarray(data).reshape([len(data), config.STAT_REPEAT, -1])
