import numpy as np
from scipy import stats
import matplotlib.pyplot
import matplotlib.ticker

from util import eval_opcodes
from util import record_load_helper
from util import count_helper
from util import stat_helper
from util import array_helper
from util import saver
from util import benchmarks
from util import tex_table
import config


def normalize(by_what, *data):
    for d in data:
        d[:] = d / by_what


def load_data(total, *line_selectors,
              data_name=config.D_RECORD_BMISS, period=config.RECORD_BMISS_PERIOD):
    dist = record_load_helper.load_line_dist(data_name, bin_name=None)
    dist = period * 1.0 * dist
    if total:
        yield dist.sum(1)
    for line_selector in line_selectors:
        yield dist[:, line_selector[:dist.shape[1]]].sum(1)


def mpkc_overall():
    selector_d = eval_opcodes.is_dispatch_line()
    miss, miss_d = load_data(True, selector_d)
    cycle, = load_data(True, data_name=config.D_RECORD_CYCLES, period=config.RECORD_CYCLE_PERIOD)

    normalize(cycle / 1000, miss, miss_d)

    fig, ax = matplotlib.pyplot.subplots(figsize=(7.5, 3))
    bm_names, = array_helper.sort_arrays(miss_d, miss, miss_d, None, benchmarks.list_names_arr())

    positions = np.arange(len(miss))
    kwargs = {'width': 1, 'edgecolor': 'k'}
    ax.bar(positions, miss - miss_d, bottom=miss_d, color='y', label='others', **kwargs)
    ax.bar(positions, miss_d, color='g', label='due to dispatch', **kwargs)

    ax.set_xticks(positions)
    ax.set_xticklabels(bm_names, rotation_mode='anchor', rotation=40, ha='right', va='center', fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel('MPKC')

    ax.margins(0.02)
    ax.legend(ncol=1, bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 'small'})
    saver.save_figure(fig, 'mpkc-overall')

    saver.save_gossip(
        'MPKC max value and that induced by dispatch',
        'max: %.1f' % miss[0],
        array_helper.summarize(miss_d, '%.1f')
    )


def mpkc_dispatch():
    cycle, = load_data(True, data_name=config.D_RECORD_CYCLES, period=config.RECORD_CYCLE_PERIOD)

    selector_d = eval_opcodes.is_dispatch_line()
    selector_d1 = eval_opcodes.dispatch_for_opcodes('LOAD_FAST')
    miss, miss_1 = load_data(False, selector_d, selector_d1)
    miss_nt, = load_data(False, selector_d, data_name=config.D_RECORD_BMISS_NT)

    opcode_count = count_helper.load(config.D_COUNT_OPCODES)[:, :-1]
    opcode_count = opcode_count.sum(1) / config.COUNT_REPEAT * config.RECORD_REPEAT
    saver.save_gossip(
        'Dispatch misprediction probability',
        'threaded: ' + array_helper.summarize(miss / opcode_count, '%.3f'),
        'non threaded: ' + array_helper.summarize(miss_nt / opcode_count, '%.3f')
    )

    normalize(cycle / 1000, miss, miss_nt, miss_1)

    def scatter_fit(x, y, color, label, **kwargs):
        k, b, r, *_ = stats.linregress(x, y)
        handles.append(ax.scatter(x, y, color=color, label=label, **kwargs))
        labels.append(label)
        label = r'r=%.3f, k=%.3f' % (r, k)
        handles.append(ax.plot(x_limit, k * x_limit + b, label=label, color=color)[0])
        labels.append(label)

    fig, ax = matplotlib.pyplot.subplots(figsize=(6, 4))
    handles = []
    labels = []
    x_limit = np.asarray([miss.min(), miss.max()])
    label = 'threaded (reference line)'
    handles.append(ax.plot(x_limit, x_limit, 'k--', label=label)[0])
    labels.append(label)
    scatter_fit(miss, miss_nt, 'darkred', 'non-threaded', marker='x', s=20)
    scatter_fit(miss, miss_1, 'limegreen', 'LOAD_FAST', marker='o', s=15)

    locator = matplotlib.ticker.MaxNLocator(integer=True)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)

    ax.legend(handles, labels, ncol=1)
    ax.margins(0.02)
    saver.save_figure(fig, 'mpkc-dispatch')


def threaded_improvement():
    bm_names = benchmarks.list_names()
    stat_cycle = stat_helper.load(config.D_STAT_MANY)[:, 1, :].sum(1)
    stat_cycle_nt = stat_helper.load(config.D_STAT_MANY_NON_THREADED)[:, 1, :].sum(1)
    reduction = (stat_cycle_nt - stat_cycle) / stat_cycle_nt
    saver.save_gossip(
        'Cycle reduction of threaded version',
        array_helper.summarize(100 * reduction, '%.1f%%'),
        'best: ' + bm_names[reduction.argmax()],
        'worst: ' + bm_names[reduction.argmin()],
    )


def dispatch_ipc():
    bm_names = benchmarks.list_names_arr()
    selector_d = eval_opcodes.is_dispatch_line()
    cycle, cycle_d = load_data(True, selector_d,
                               data_name=config.D_RECORD_CYCLES, period=config.RECORD_CYCLE_PERIOD)
    instr, instr_d = load_data(True, selector_d,
                               data_name=config.D_RECORD_INSTR, period=config.RECORD_INSTR_PERIOD)
    ipc = instr / cycle * (config.RECORD_INSTR_PERIOD / config.RECORD_CYCLE_PERIOD)
    ipc_d = instr_d / cycle_d * (config.RECORD_INSTR_PERIOD / config.RECORD_CYCLE_PERIOD)
    bm_names, = array_helper.sort_arrays(ipc, ipc, ipc_d, None, bm_names)

    saver.save_gossip(
        'Overhead for dispatch',
        'overall: ' + array_helper.summarize(cycle_d / cycle * 100, '%.1f')
    )
    p_value = stats.ttest_rel(ipc, ipc_d)[1]

    positions = np.arange(len(ipc))
    fig, ax = matplotlib.pyplot.subplots(figsize=(7.5, 3))
    ax.plot(positions, ipc, 'b', label='the whole program', marker='.', markersize=4)
    ax.scatter(positions, ipc_d, 15, color='#11dd00', marker='o',
               label=r'dispatch (p=%s)' % tex_table.math_number('%#.3g', p_value))

    ax.set_xticks(positions)
    ax.set_xticklabels(bm_names, rotation_mode='anchor', rotation=40, ha='right', va='center', fontsize=8)
    ax.set_ylabel(r'IPC (Instructions Per Cycle)')
    ax.legend(fontsize='small')
    ax.margins(0.02)

    saver.save_figure(fig, 'dispatch-ipc')


if __name__ == '__main__':
    mpkc_overall()
    mpkc_dispatch()
    threaded_improvement()
    dispatch_ipc()
