import numpy as np
import matplotlib.pyplot
import matplotlib.ticker

from util import record_load_helper
from util import count_helper
from util import hsy_color
from util import saver
from util import array_helper
from util import benchmarks
from util import eval_opcodes
import config


def stacked_hbar(dist, selector, labels, colors, legend_prop, save_name):
    dist = dist[selector]
    bm_names = benchmarks.list_names_arr()[selector]
    fig, ax = matplotlib.pyplot.subplots(figsize=(7, 3))
    positions = np.arange(len(bm_names))

    left = np.zeros_like(dist[:, 0])
    for d, c, l in zip(dist.T, colors, labels):
        ax.barh(positions, d, left=left, label=l, color=c)
        left += d
    for x, y in zip(left, positions):
        ax.text(x, y, ' %.1f%%' % x, va='center')

    ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1,
              prop=legend_prop)
    ax.xaxis.tick_top()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, 0, '%'))
    ax.set_xlim(0, dist[-1].sum() * 1.12)
    ax.set_yticks(positions)
    ax.set_yticklabels(bm_names)
    ax.margins(0.02)
    saver.save_figure(fig, save_name)


def math_operator():
    dist = record_load_helper.load_line_dist_cg(config.D_RECORD_CYCLES_CG)
    opcode_remap_arr, opcode_names = eval_opcodes.line_to_opcode()

    category_remap_arr, category_names = eval_opcodes.opcode_to_category()
    is_math_op = category_remap_arr[opcode_remap_arr] == list(category_names).index('math operator')
    is_math_op = is_math_op & (opcode_remap_arr != -1)

    stmt_remap_arr, stmt_names = eval_opcodes.line_to_stmt()
    is_misc = stmt_remap_arr == list(stmt_names).index('miscellany')

    dist = dist / dist[:, 0, :].sum(-1)[:, None, None] * 100
    dist_ = np.empty([len(dist), 3], dtype=dist.dtype)
    dist_[:, 0] = dist[:, 0, is_math_op & ~is_misc].sum(-1) + dist[:, 3, is_math_op & is_misc].sum(-1)
    dist_[:, 2] = dist[:, 2, is_math_op & is_misc].sum(-1)
    dist_[:, 1] = dist[:, 0, is_math_op].sum(-1) - dist_[:, 0] - dist_[:, 2]
    dist = dist_

    labels = ['opcode handling', 'overloading', 'calculation']
    selector = dist.sum(1).argsort()[-10:]
    colors = hsy_color.hsy_to_rgb([0.3, 0.6, 0.22], 1, [0.4, 0.6, 0.8])
    stacked_hbar(dist, selector, labels, colors, {}, 'math-operator')

    saver.save_gossip(
        'operator overloading',
        array_helper.summarize(dist[:, 1], '%.1f')
    )


def attr_access():
    dist, opcode_names = record_load_helper.load_opcode_dist_cg()
    dist = array_helper.normalize(dist) * 100

    labels = ['LOAD_ATTR', 'LOAD_METHOD', 'STORE_ATTR', 'DELETE_ATTR']
    dist = dist[:, [np.where(opcode_names == o)[0][0] for o in labels]]
    selector = dist.sum(1).argsort()[-10:]
    colors = hsy_color.hsy_to_rgb([0.3, 0.6, 0.22, 0], 1, [0.4, 0.6, 0.8, 0.1])
    stacked_hbar(dist, selector, labels, colors, {'family': 'monospace'}, 'attr-access')

    saver.save_gossip(
        'attribute access overhead',
        array_helper.summarize(dist.sum(1), '%.1f')
    )


def atomic_type():
    dist = count_helper.load(config.D_COUNT_ATOMIC_TYPES)
    # [benchmarks, (operator, LOAD_ATTR, LOAD_METHOD), (not_atomic, is_atomic)]
    dist = dist.reshape([-1, 3, 2])[:, 1:, :]
    dist_sum = dist.sum((1, 2))
    attr_ratio_dist = dist[:, 0, 1] / dist_sum
    method_ratio_dist = dist[:, 1, 1] / dist_sum

    dist, opcode_names = record_load_helper.load_opcode_dist_cg()
    dist = array_helper.normalize(dist) * 100
    attr_overhead_dist = dist[:, np.argmax(opcode_names == 'LOAD_ATTR')]
    method_overhead_dist = dist[:, np.argmax(opcode_names == 'LOAD_METHOD')]

    bm_names, = array_helper.sort_arrays(attr_overhead_dist + method_overhead_dist,
                                         attr_overhead_dist, method_overhead_dist, attr_ratio_dist, method_ratio_dist,
                                         None, benchmarks.list_names_arr())

    fig, ax = matplotlib.pyplot.subplots(figsize=(7.5, 3.5))
    positions = np.arange(len(method_overhead_dist))

    colors = hsy_color.hsy_to_rgb([0.4, 0.95], 1, [0.8, 0.2])
    kwargs = {'width': 1, 'linewidth': 0.2}
    m1 = ax.bar(positions, method_ratio_dist * method_overhead_dist, color=colors[0], edgecolor='none', **kwargs)
    m0 = ax.bar(positions, method_overhead_dist, color='none', edgecolor='k', **kwargs)
    a1 = ax.bar(positions, -attr_ratio_dist * attr_overhead_dist, color=colors[1], edgecolor='none', **kwargs)
    a0 = ax.bar(positions, -attr_overhead_dist, color='none', edgecolor='k', **kwargs)

    legend1 = ax.legend([m0, m1], ['LOAD_METHOD', 'LOAD_METHOD for atomic types'],
                        ncol=1, bbox_to_anchor=(1, 1), loc='upper right')
    legend2 = ax.legend([a1, a0], ['LOAD_ATTR for atomic types', 'LOAD_ATTR'],
                        ncol=1, bbox_to_anchor=(1, 0), loc='lower right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_xticks(positions)
    ax.set_xticklabels(bm_names, rotation_mode='anchor', rotation=40, ha='right', va='center', fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(lambda tick, pos: str(int(abs(tick))))
    ax.set_ylabel('Overhead (%)')

    ax.margins(0.01)
    saver.save_figure(fig, 'atomic-types')

    potential = attr_ratio_dist * attr_overhead_dist + method_ratio_dist * method_overhead_dist
    saver.save_gossip(
        'atomic types',
        array_helper.summarize(potential, '%.1f%%'),
        array_helper.summarize(potential, '%.3f%%'),
        'the max value is on benchmark: %s' % bm_names[potential.argmax()],
        'the 2nd max value is: %.1f%%\n' % potential[potential.argsort()[-2]]
    )


if __name__ == '__main__':
    math_operator()
    attr_access()
    atomic_type()
