from util import count_helper
from util import record_collect_helper
from util import stat_helper
import config

record_collect_helper.collect(config.D_RECORD_CYCLES, 'cycles:P', config.RECORD_CYCLE_PERIOD)
record_collect_helper.collect(config.D_RECORD_CYCLES_CG, 'cycles', config.RECORD_CYCLE_PERIOD, call_graph=2048)
record_collect_helper.collect(config.D_RECORD_INSTR, 'instructions:P', config.RECORD_INSTR_PERIOD)
record_collect_helper.collect(config.D_RECORD_BMISS, 'branch-misses:P', config.RECORD_BMISS_PERIOD)

record_collect_helper.collect(config.D_RECORD_BMISS_NT, 'branch-misses:P', config.RECORD_BMISS_PERIOD,
                              cpython=config.cpy_exe_path('non-threaded'))

count_helper.collect(config.D_COUNT_OPCODES)
count_helper.collect(config.D_COUNT_ATOMIC_TYPES)
count_helper.collect(config.D_COUNT_PREDEFINED)

stat_helper.collect(config.D_STAT_MANY, 'instructions,cycles,branch-misses')
stat_helper.collect(config.D_STAT_MANY_NON_THREADED, 'instructions,cycles,branch-misses',
                    cpython=config.cpy_exe_path('non-threaded'))
