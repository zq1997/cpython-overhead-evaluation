import numpy as np
import matplotlib.pyplot
import matplotlib.ticker

from util import record_load_helper
from util import count_helper
from util import eval_opcodes
from util import saver
from util import array_helper
import config


def boxplot(freq_dist, overhead_dist, figsize, legend_kwargs, tick_labels):
    fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
    positions = np.arange(len(freq_dist[0]))
    width = 0.35

    params1 = (freq_dist, '#00dd66', 0, 'frequency')
    params2 = (overhead_dist, '#bb0000', 1, 'overhead')
    to_legend = []
    for dist, color, offset, label in [params1, params2]:
        bp = ax.boxplot(dist,
                        positions=positions + offset * width - width / 2,
                        medianprops=dict(color='black'),
                        patch_artist=True,
                        sym='',
                        widths=width - 0.05)

        to_legend.append((bp["boxes"][0], label))
        for patch in bp['boxes']:
            patch.set(facecolor=color)

    ax.set_xticks(positions)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, '%'))
    ax.legend(*zip(*to_legend), **legend_kwargs)
    ax.set_xticklabels(tick_labels,
                       rotation_mode='anchor', rotation=15, ha='right', va='center',
                       family='monospace', fontsize=8)
    return fig, ax


def name_access():
    def dist_of_names(dist, names):
        names = {n: i for i, n in enumerate(names)}
        seletion_schema = [
            'LOAD_CONST',
            'LOAD_FAST',
            'STORE_FAST',
            'DELETE_FAST',
            ['*_DEREF', 'LOAD_DEREF', 'STORE_DEREF', 'DELETE_DEREF'],
            'LOAD_CLASSDEREF',
            'LOAD_GLOBAL',
            'STORE_GLOBAL',
            'DELETE_GLOBAL',
            ['*_NAME', 'LOAD_NAME', 'STORE_NAME', 'DELETE_NAME']
        ]

        selected_names = []
        selected_dist = []
        for sel in seletion_schema:
            if isinstance(sel, str):
                selected_names.append(sel)
                selected_dist.append(dist[:, names[sel]][..., None])
            else:
                selected_names.append(sel[0])
                selected_dist.append(dist[:, [names[n] for n in sel[1:]]].sum(-1)[..., None])
        selected_dist = np.concatenate(selected_dist, axis=1)
        dist_sum = dist.sum(1, keepdims=True)
        return selected_names, selected_dist, selected_dist / dist_sum

    count_dist = count_helper.load(config.D_COUNT_OPCODES)[:, :-1]
    count_dist = count_dist * (config.RECORD_REPEAT / config.COUNT_REPEAT)
    opcode_names = eval_opcodes.get_opcode_names()
    names, count_dist, freq_dist = dist_of_names(count_dist, opcode_names)

    cycle_dist, opcode_names = record_load_helper.load_opcode_dist_cg()
    cycle_dist = config.RECORD_CYCLE_PERIOD * cycle_dist
    _, cycle_dist, overhead_dist = dist_of_names(cycle_dist, opcode_names)

    fig, ax = boxplot(
        freq_dist,
        overhead_dist,
        (8.5, 3),
        dict(loc='upper center', framealpha=1),
        names
    )

    ax.axvline(4.5, linestyle='--', color='k')
    ax.text(4.25, 0.3, 'array-style', ha='right')
    ax.text(4.75, 0.3, 'dict-style', ha='left')
    saver.save_figure(fig, 'name-access')

    saver.save_gossip(
        'name access opcodes ([frequence, overhead] x [array-style, dict-style])',
        array_helper.summarize(freq_dist[:, :5].sum(-1) * 100, '%.1f'),
        array_helper.summarize(freq_dist[:, 5:].sum(-1) * 100, '%.1f'),
        array_helper.summarize(overhead_dist[:, :5].sum(-1) * 100, '%.1f'),
        array_helper.summarize(overhead_dist[:, 5:].sum(-1) * 100, '%.1f')
    )
    array_cpo_dist = cycle_dist[:, 1] / count_dist[:, 1]
    dict_cpo_dist = cycle_dist[:, 6] / count_dist[:, 6]
    saver.save_gossip(
        'cycle per opcode (LOAD_FAST vs LOAD_GLOBAL)',
        array_helper.summarize(array_cpo_dist, '%.1f'),
        array_helper.summarize(dict_cpo_dist, '%.1f')
    )


def load_global_opcache():
    opcode_names = eval_opcodes.get_opcode_names()
    count_dist = count_helper.load(config.D_COUNT_OPCODES)
    count_dist = count_dist * (config.RECORD_REPEAT / config.COUNT_REPEAT)
    count_sum = count_dist.sum(1, keepdims=True)
    count_lg = np.empty([len(count_dist), 3], dtype=count_dist.dtype)
    count_lg[:, 2] = count_dist[:, :-1][:, opcode_names == 'LOAD_GLOBAL'][:, 0]
    count_lg[:, 0] = count_dist[:, -1]
    count_lg[:, 1] = count_lg[:, 2] - count_lg[:, 0]
    freq_lg = count_lg / count_sum
    hit_ratio = count_lg[:, 0] / count_lg[:, 2]

    cycle_dist = record_load_helper.load_line_dist_cg(config.D_RECORD_CYCLES_CG)[:, 0]
    cycle_dist = config.RECORD_CYCLE_PERIOD * cycle_dist
    remap_arr = eval_opcodes.load_global_lines()
    cycle_dist = array_helper.remap_dist_by_arr(cycle_dist, remap_arr, 5)
    cycle_sum = cycle_dist.sum(1, keepdims=True)
    cycle_lg = np.empty([len(cycle_dist), 3], dtype=cycle_dist.dtype)
    cycle_lg[:, 2] = cycle_dist[:, 1:].sum(1)
    cycle_lg[:, 0] = cycle_dist[:, 1] + cycle_dist[:, 2] + cycle_dist[:, 3] * hit_ratio
    cycle_lg[:, 1] = cycle_lg[:, 2] - cycle_lg[:, 0]
    overhead_lg = cycle_lg / cycle_sum

    fig, ax = boxplot(
        freq_lg,
        overhead_lg,
        (3.5, 3),
        dict(loc='upper center', framealpha=1),
        ['LOAD_GLOBAL(hit)', 'LOAD_GLOBAL(miss)', 'LOAD_GLOBAL']
    )
    saver.save_figure(fig, 'load-global-opcache')

    cpo = cycle_lg / count_lg
    saver.save_gossip(
        'per-opcode cache (hit ratio and cycle per opcode when miss)',
        array_helper.summarize(hit_ratio * 100, '%.1f'),
        array_helper.summarize(cpo[:, 1], '%.1f')
    )


if __name__ == '__main__':
    name_access()
    load_global_opcache()
