import matplotlib.pyplot
import matplotlib.ticker
import numpy as np

from util import eval_opcodes
from util import saver
from util import record_load_helper
from util import array_helper
from util import count_helper
from util import tex_table
import config


def dump_statistic_table(save_name, dist, labels):
    dist = dist * 100

    dist_ = dist[:, :eval_opcodes.OPCODE_NUMBER]
    labels_ = labels[:eval_opcodes.OPCODE_NUMBER]
    labels_, = array_helper.sort_arrays(np.median(dist_, 0), dist_.T, None, labels_)

    cut_num = 20
    other_dist = np.sum(dist_[:, cut_num:], 1, keepdims=True)
    dist = np.concatenate([dist_[:, :cut_num], other_dist, dist[:, eval_opcodes.OPCODE_NUMBER:]], axis=1)
    labels = [*labels_[:cut_num], '<OTHER_OPCODES>', *labels[eval_opcodes.OPCODE_NUMBER:]]

    quartile1, quartile2, quartile3 = np.percentile(dist, [25, 50, 75], 0)

    table = tex_table.LatexTable()
    table.append_column(
        *tex_table.emcell('rank'),
        *[str(i + 1) if i < cut_num else '' for i in range(len(dist.T))],
        style=r'r'
    )
    table.append_column(
        *tex_table.emcell('opcode'),
        *tex_table.bracket(*labels, cmd='code'),
        style=r'l'
    )

    def append_data_column(nums, head, space=False):
        table.append_column(
            *tex_table.emcell(head),
            *('%.1f' % n for n in nums),
            style=r'@{\hspace{%d\tabcolsep}}S' % (4 if space else 1)
        )

    append_data_column(quartile1, '$Q_1$', space=True)
    append_data_column(quartile2, '$Q_2$')
    append_data_column(quartile3, '$Q_3$')
    append_data_column(np.mean(dist, 0), 'mean', space=True)
    append_data_column(np.std(dist, 0), 'std')

    saver.save_table(table, save_name)


def opcode_frequency_dist():
    dist = count_helper.load(config.D_COUNT_OPCODES)[:, :-1]
    dist = array_helper.normalize(dist)
    opcode_names = eval_opcodes.get_opcode_names()
    dump_statistic_table('opcode-frequency', dist, opcode_names)


def opcode_overhead_dist():
    dist, opcode_names = record_load_helper.load_opcode_dist_cg()
    dist = array_helper.normalize(dist)
    dump_statistic_table('opcode-overhead', dist, opcode_names)


def category_dist():
    remap_arr, category_names = eval_opcodes.opcode_to_category()
    dist1 = count_helper.load(config.D_COUNT_OPCODES)[:, :-1]
    dist1 = array_helper.normalize(dist1)

    dist2, _ = record_load_helper.load_opcode_dist_cg()
    dist2 = array_helper.normalize(dist2)
    dist2 = dist2[:, :eval_opcodes.OPCODE_NUMBER]

    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 3))
    positions = np.arange(len(category_names))
    width = 0.35
    anno_attrs = {'xytext': (0, -50), 'textcoords': 'offset points', 'fontsize': 'small'}
    ax.annotate('$Q_1$\n$Q_2$\n$Q_3$', (0, 0),
                fontsize='small', textcoords='offset points', xytext=(-50, -50))

    params1 = (dist1, '#00dd66', 0, 'frequency')
    params2 = (dist2, '#bb0000', 1, 'overhead')
    to_legend = []
    for dist, color, offset, label in [params1, params2]:
        dist = array_helper.remap_dist_by_arr(dist * 100, remap_arr, len(category_names))
        q1, q2, q3 = np.quantile(dist, [0.25, 0.5, 0.75], 0)
        pos = positions + offset * width - width / 2

        bp = ax.boxplot(dist, positions=pos,
                        flierprops=dict(markersize=5, markerfacecolor=color),
                        medianprops=dict(color='black'),
                        patch_artist=True,
                        widths=width - 0.03)

        to_legend.append((bp["boxes"][0], label))
        for patch in bp['boxes']:
            patch.set(facecolor=color)

        for i in range(len(pos)):
            ax.annotate('%.1f%%\n%.1f%%\n%.1f%%' % (q1[i], q2[i], q3[i]),
                        (pos[i], 0), ha='center', color=color, **anno_attrs)

    ax.set_xticks(positions)
    ax.set_xticklabels(category_names, family='serif', style='italic', fontsize='small')
    ax.tick_params(axis='x', length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, 0, '%'))
    ax.legend(*zip(*to_legend), loc='upper right')
    saver.save_figure(fig, 'opcode-category')


if __name__ == '__main__':
    opcode_frequency_dist()
    opcode_overhead_dist()
    category_dist()
