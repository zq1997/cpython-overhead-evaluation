import subprocess
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot
import matplotlib.ticker

from util import record_load_helper
from util import count_helper
from util import saver
from util import tex_table
from util import benchmarks
from util import log
from util import array_helper
import config


def get_counting_dist(data_name, inc=True, dec=True):
    return record_load_helper.load_line_dist(
        data_name=data_name,
        bin_name=None,
        src_name=[
            os.path.join(config.CPY_SRC_DIR, 'Include', 'object.h'),
            os.path.join(config.CPY_INSTALL_DIR, 'include', 'python3.9', 'object.h')
        ],
        func_name=(*(('_Py_INCREF',) * inc), *(('_Py_DECREF',) * dec)),
        inline=True, max_line=1)


def gc_comparison():
    counting_dist = get_counting_dist(config.D_RECORD_CYCLES)
    tracing_dist = record_load_helper.load_line_dist(
        bin_name=None,
        src_name=os.path.join(config.CPY_SRC_DIR, 'Modules', 'gcmodule.c'),
        func_name=None, max_line=1
    )
    perf_count = counting_dist.sum(1)
    counting_dist = counting_dist[:, 1] / perf_count
    tracing_dist = tracing_dist[:, 1] / perf_count

    opcode_dist = count_helper.load(config.D_COUNT_OPCODES)[:, :-1]
    lopc_dist = np.log2(
        (opcode_dist.sum(1) / config.COUNT_REPEAT) /
        (perf_count * config.RECORD_CYCLE_PERIOD / config.RECORD_REPEAT)
    )

    x_limit = np.asarray([lopc_dist.min(), lopc_dist.max()])

    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3.5))
    table = tex_table.LatexTable()
    table.append_column('', '$Q_1$', '$Q_2$', '$Q_3$', 'Pearson\'s $r$')

    color = 'darkred'
    ax.scatter(lopc_dist, counting_dist, color=color, marker='.', label='counting')
    slope, b, r, *_ = stats.linregress(lopc_dist, counting_dist)
    ax.plot(x_limit, slope * x_limit + b, color=color)
    table.append_column(
        *tex_table.emcell('counting'),
        *(r'\SI{%.1f}{\%%}' % (q * 100) for q in np.quantile(counting_dist, (0.25, 0.5, 0.75), 0)),
        '%.3f' % r,
    )

    color = 'limegreen'
    ax.scatter(lopc_dist, tracing_dist, color=color, marker='1', label='tracing')
    r = stats.pearsonr(lopc_dist, tracing_dist)[0]
    q1, q2, q3 = np.quantile(tracing_dist, (0.25, 0.5, 0.75), 0)
    ax.plot(x_limit, [q2, q2], color=color)
    table.append_column(
        *tex_table.emcell('tracing'),
        *(r'\SI{%.1f}{\%%}' % (q * 100) for q in (q1, q2, q3)),
        '%.3f' % r,
    )

    bm_names = benchmarks.list_names()
    indexes = [bm_names.index(x) for x in ['python_startup', 'python_startup_no_site']]
    indexes.sort(key=lambda i: lopc_dist[i])
    vas = ['bottom', 'top']
    for i, va in zip(indexes, vas):
        ax.annotate(bm_names[i], (lopc_dist[i], tracing_dist[i]),
                    rotation_mode='anchor', rotation=12, ha='left', va=va, color='gray', zorder=0)

    ax.legend()
    ax.set_xlabel(r'$LOPC=\log_2(\frac{N_{opcodes}}{N_{cycles}})$')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, '%'))
    ax.margins(0.02)

    saver.save_figure(fig, 'gc-comparison')
    saver.save_table(table, 'gc-comparison')

    saver.save_gossip(
        'LOPC slope',
        '%.1f%%' % (slope * 100)
    )


@saver.save_result_as_pickle('gc-threshold')
def stat_gc_threshold():
    def run_cmd(cmd_prefix, args, env, stdout_result):
        warm_repeat = 500
        run_repeat = 5000
        stdout_list = []
        for i in range(-warm_repeat, run_repeat):
            proc = subprocess.Popen(
                [*cmd_prefix, config.cpy_exe_path('gc-threshold'), *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
                universal_newlines=True,
            )
            if i >= 0:
                stdout_list.append(stdout_result(proc.stderr.read()))
            assert proc.wait() == 0
        return stdout_list

    # return array(N * 3), N runs, 3 for (instructions, cycles, maxRSS)
    def stat_single(env, args):
        data1 = run_cmd(
            ['perf', 'stat', '-x,', '-e', 'instructions,cycles'],
            args, env,
            lambda s: [int(x.split(',', 1)[0]) for x in s.splitlines()]
        )
        data2 = run_cmd(
            ['time', '-f', '%M'],
            args, env,
            lambda s: [int(s.rstrip())]
        )
        return np.hstack((data1, data2))

    # return array(K * N * 3), K for different thresholds
    def stat_multiple(*args):
        data = []
        log.message('python %s:' % ' '.join(args))
        for factor in gc_factors:
            log.message(' *%d' % factor)
            env = {'GC_THRESHOLD_FACTOR': str(factor)}
            data.append(stat_single(env, args))
        log.message('\n')
        return np.asarray(data)

    log.title('gc-threshold')
    gc_factors = (1, 2, 4, 8, 16, 32, 2 ** 20)
    return gc_factors, stat_multiple('-c', 'pass'), stat_multiple('-S', '-c', 'pass')


def gc_threshold():
    gc_factors, *dists = stat_gc_threshold()
    names = ['gc-threshold-normal', 'gc-threshold-no-site']

    positions = [*range(6), 7]
    tick_labels = [1, *('$2^{%d}$' % i for i in range(1, 6)), '$2^{20}$']
    ellipsis_position = 6

    for name, dist in zip(names, dists):
        dist = dist.mean(1)
        dist_base = dist[0]
        dist = dist / dist_base

        fig, ax = matplotlib.pyplot.subplots(figsize=(5, 2.5))

        labels = [
            'instructions (%s)' % tex_table.math_number('%#.2g', dist_base[0]),
            'CPU cycles (%s)' % tex_table.math_number('%#.2g', dist_base[1]),
            'maximum RSS (%.1f MB)' % (dist_base[2] / 2 ** 10)
        ]
        line_types = ['-x', '-o', '-d']
        for i in range(dist.shape[1]):
            ax.plot(positions, dist[:, i], line_types[i], label=labels[i])

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xticks([ellipsis_position], minor=True)
        ax.set_xticklabels(['...'], minor=True, fontsize='large')
        ax.set_yticks([x / 100 for x in range(97, 101)])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, '%'))
        ax.tick_params(axis='x', which='minor', length=0)

        ax.legend(loc='right', fontsize='small')
        saver.save_figure(fig, name)

        saver.save_gossip(
            'improvement: ' + name,
            ' '.join(['%.1f%%' % x for x in 100 - 100 * dist[:, 1]])
        )


def ipc_for_counting():
    def get_dist(dataname):
        inc_dist = get_counting_dist(dataname, dec=False)
        dec_dist = get_counting_dist(dataname, inc=False)
        return np.vstack((inc_dist.sum(1), inc_dist[:, 1], dec_dist[:, 1])).T

    cycle_dist = get_dist(config.D_RECORD_CYCLES)
    instr_dist = get_dist(config.D_RECORD_INSTR)

    dist = instr_dist / cycle_dist * (config.RECORD_INSTR_PERIOD / config.RECORD_CYCLE_PERIOD)
    bm_names, = array_helper.sort_arrays(dist[:, 0], dist, None, benchmarks.list_names_arr())

    def plot(i, color, label, marker):
        p = tex_table.math_number('%#.3g', stats.ttest_rel(dist[:, 0], dist[:, i])[1])
        ax.scatter(positions, dist[:, i], 15, color=color, marker=marker,
                   label=r'$\mathtt{%s}$ (p=%s)' % (label, p))

    positions = np.arange(len(dist))
    fig, ax = matplotlib.pyplot.subplots(figsize=(7.5, 3))
    ax.plot(positions, dist[:, 0], 'b', label='the whole program', marker='.', markersize=4)
    plot(1, '#11dd00', r'\_Py\_INCREF', 'o')
    plot(2, '#990011', r'\_Py\_DECREF', 'x')

    ax.set_xticks(positions)
    ax.set_xticklabels(bm_names, rotation_mode='anchor', rotation=40, ha='right', va='center', fontsize=8)
    ax.set_ylabel(r'IPC (Instructions Per Cycle)')
    ax.legend(fontsize='small')
    ax.margins(0.02)
    saver.save_figure(fig, 'counting-ipc')

    saver.save_gossip(
        'IPC of _Py_INCREF is reduced by',
        array_helper.summarize(100 * (1 - dist[:, 1] / dist[:, 0]), '%.1f%%')
    )


if __name__ == '__main__':
    gc_comparison()
    gc_threshold()
    ipc_for_counting()
