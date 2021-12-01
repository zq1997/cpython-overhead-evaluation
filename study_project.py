import os

import numpy as np
import matplotlib.pyplot
import matplotlib.ticker

from util import hsy_color
from util import record_load_helper
from util import array_helper
from util import eval_opcodes
from util import saver
from util import benchmarks
import config


def violin_plot(save_name, dist, labels, figsize, color, detail):
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=figsize)
    positions = np.arange(len(labels)) + 1
    q1, q2, q3 = np.percentile(dist, [25, 50, 75], 0)

    for ax in (ax1, ax2):
        parts = ax.violinplot(dist, showextrema=False, vert=False, widths=0.9, bw_method=0.3)
        for p in parts['bodies']:
            p.set_alpha(1)
            p.set_color(color)

        ax.scatter(dist.T.flatten(), positions.repeat(len(dist)), marker='.', color='k', s=3, zorder=2)
        ax.hlines(positions, q1, q3, color='k', lw=4)
        ax.scatter(q2, positions, marker='o', color='w', s=15, zorder=3)

        ax.xaxis.tick_top()
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, '%'))
        ax.tick_params('both', pad=0)
        ax.margins(0.01)
        ax.invert_yaxis()
    ax1.set_xlim(0, detail)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(['%s\n%.1f%%' % x for x in zip(labels, q2 * 100)], family='monospace', fontsize='small')
    ax2.set_yticks([])
    saver.save_figure(fig, save_name)


def bin_dist():
    bm_names = benchmarks.list_names_arr()

    dist, bin_names = record_load_helper.load_name_dist('bin')
    remap_arr = np.full_like(bin_names, 2, dtype=np.int)
    remap_arr[[x.startswith('[') for x in bin_names]] = 3
    remap_arr[[x.startswith(config.CPY_INSTALL_DIR) for x in bin_names]] = 1
    remap_arr[bin_names == config.CPY_EXE] = 0
    dist = array_helper.normalize(array_helper.remap_dist_by_arr(dist, remap_arr, len(bin_names)))

    func_dist = record_load_helper.load_line_dist(max_line=1)
    func_dist = array_helper.normalize(func_dist)[:, 1]

    array_helper.sort_arrays(dist[:, 0], dist, func_dist, bm_names)

    labels = [
        'main executable',
        'built-in libraries',
        'external libraries',
        'kernel space'
    ]
    fig, ax = matplotlib.pyplot.subplots(figsize=(8.5, 3.5))
    colors = hsy_color.hsy_to_rgb([0, 0.3, 0.6, 0.22], 1, [0.1, 0.4, 0.7, 0.9])
    positions = np.arange(dist.shape[0])
    bottom = np.zeros_like(dist[:, 0])
    for d, c, l in zip(dist.T, colors, labels):
        ax.bar(positions, d, width=0.9, bottom=bottom, label=l, color=c)
        bottom += d
    ax.bar(positions, func_dist, width=0.9, facecolor='none', lw=0, edgecolor='#ff1111', hatch='xxx')
    ax.set_xticks(positions)
    ax.set_xticklabels(bm_names, rotation_mode='anchor', rotation=40, ha='right', va='center', fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, '%'))
    ax.margins(x=0.005, y=0)
    ax.legend(bbox_to_anchor=(1, 1), fontsize='small', loc='lower right', ncol=len(labels))
    saver.save_figure(fig, 'bin')


def src_dist():
    dist, src_names = record_load_helper.load_name_dist('src')
    dist = array_helper.normalize(dist)
    median = np.median(dist, 0)
    src_names, = array_helper.sort_arrays(median, median, dist.T, None, src_names)
    selector = (median >= 0.001) & (src_names != '??')
    dist = np.append(dist[:, selector], np.sum(dist[:, ~selector], 1, keepdims=True), 1)
    src_names = [os.path.relpath(s, config.CPY_SRC_DIR) for s in src_names[selector]]
    src_names = (*src_names, '<OTHER_FILES>')
    root = config.CPY_EXE
    src_names = np.asarray([p[len(root):] if p.startswith(root) else p for p in src_names])
    violin_plot('src', dist, src_names, (9, 5), 'deepskyblue', 0.107)


def func_dist(inline):
    dist, func_names = record_load_helper.load_name_dist('func', inline=inline)
    dist = array_helper.normalize(dist)
    func_names, = array_helper.sort_arrays(np.median(dist, 0), dist.T, None, func_names)
    dist = dist[:, :10]
    func_names = func_names[:10]
    save_name = ('i' if inline else 'n') + 'func'
    color = 'teal' if inline else 'goldenrod'
    violin_plot(save_name, dist, func_names, (8, 3), color, 0.087)


def stmt_dist():
    dist = record_load_helper.load_line_dist()
    dist[:, 0] = 0
    dist = array_helper.normalize(dist)
    remap_arr, stmt_names = eval_opcodes.line_to_stmt()
    dist = array_helper.remap_dist_by_arr(dist, remap_arr, len(stmt_names))
    median = np.median(dist, 0)
    stmt_names, = array_helper.sort_arrays(median, median, dist.T, None, stmt_names)

    fig, ax = matplotlib.pyplot.subplots(figsize=(4.5, 2.5))
    positions = np.arange(len(stmt_names)) + 1
    q1, q2, q3 = np.percentile(dist, [25, 50, 75], 0)

    parts = ax.violinplot(dist, showextrema=False, vert=False, widths=0.8, bw_method=0.3)
    for p in parts['bodies']:
        p.set_alpha(1)
        p.set_color('violet')

    ax.scatter(dist.T.flatten(), positions.repeat(len(dist)), marker='.', color='k', s=3, zorder=2)
    ax.hlines(positions, q1, q3, color='k', lw=4)
    ax.scatter(q2, positions, marker='o', color='w', s=15, zorder=3)

    ax.xaxis.tick_top()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, '%'))
    ax.tick_params('both', pad=0)
    ax.margins(0.01)
    ax.invert_yaxis()
    ax.set_yticks(positions)
    ax.set_yticklabels(['%s\n%.1f%%' % x for x in zip(stmt_names, q2 * 100)], fontsize='small')
    saver.save_figure(fig, 'stmt')


if __name__ == '__main__':
    bin_dist()
    src_dist()
    func_dist(False)
    func_dist(True)
    stmt_dist()
