import os

import psutil


def conf_path(*paths):
    return os.path.normpath(os.path.join(SELF_DIR, *paths))


def cpy_install_path(build_name):
    return os.path.join(CPY_BUILD_DIR, 'install-' + build_name)


def cpy_exe_path(build_name):
    return os.path.join(cpy_install_path(build_name), 'bin', 'python3.9')


def data_path(*paths):
    return os.path.join(DATA_DIR, *paths)


SELF_DIR = os.path.dirname(__file__)
DATA_DIR = conf_path('data')
CPY_SRC_DIR = conf_path(os.path.pardir, 'cpython')
CPY_BUILD_DIR = conf_path(os.path.pardir, 'build')
PIP_DIR = conf_path(os.path.pardir, 'benchmarks')
PAPER_DIR = conf_path(os.path.pardir, 'paper')

PYPERFORMANCE_VERSION = '1.0.1'
CALIBRATE_DIR = data_path('calibrate')
CALIBRATE_JSON = data_path('calibrate.json')
CPY_FUNC_JSON = data_path('cpython-functions.json')
NATIVE_FUNC_JSON = data_path('native-python-functions.json')

CPY_VERSION = 'v3.9.0'
CPY_INSTALL_DIR = cpy_install_path('release')
CPY_EXE = cpy_exe_path('release')

PARALLEL = max(psutil.cpu_count(logical=False) - 1, 1)
TIME_SCALE = 5100 / psutil.cpu_freq().max
RECORD_CYCLE_PERIOD = 5000011
RECORD_INSTR_PERIOD = 13000027
RECORD_BMISS_PERIOD = 40009
WARM_UP_DURATION = 5 * TIME_SCALE
RUN_DURATION = 10 * TIME_SCALE
RECORD_REPEAT = 40
STAT_REPEAT = 5
COUNT_REPEAT = 5

D_RECORD_CYCLES = 'record-cycles'
D_RECORD_CYCLES_CG = 'record-cycles-cg'
D_RECORD_INSTR = 'record-instructions'
D_RECORD_BMISS = 'record-branch-misses'
D_RECORD_BMISS_NT = 'record-branch-misses-non-threaded'
D_COUNT_OPCODES = 'count-opcodes'
D_COUNT_ATOMIC_TYPES = 'count-atomic-types'
D_STAT_MANY = 'stat-many'
D_STAT_MANY_NON_THREADED = 'stat-many-non-threaded'
