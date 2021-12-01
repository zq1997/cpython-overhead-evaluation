import numpy as np

from util import tex_table
from util import saver
from util import record_load_helper
import config


def estimate_confidence():
    n = np.median(record_load_helper.load_name_dist('bin')[0].sum(1))
    n = np.round(n, -2)
    saver.save_gossip('Sample size', '%.1e' % n)

    r = config.RECORD_CYCLE_PERIOD

    errors = (0.001, 0.002, 0.005, 0.010, 0.020)
    true_values = (0.01, 0.05, 0.10, 0.25)

    table = tex_table.LatexTable()
    table.append_row(
        *tex_table.bracket(*tex_table.emcell(r'$H$'), cmd=r'multirow{2}*[-\aboverulesep]'),
        *tex_table.bracket(*tex_table.emcell(r'$\alpha$'), cmd='multicolumn{5}{c}')
    )
    table.set_hline(1, r'\cmidrule(lr){2-6}')
    table.append_row(*tex_table.emcell(
        '',
        *(r'$\epsilon = %.1f\%%$' % (e * 100) for e in errors)
    ))
    table.set_hline(2)

    for h in true_values:
        var = h * (1 - h) * ((r - 1) / (r * n - 1))
        row = [r'$%.1f\%%$' % (h * 100)]
        for epsilon in errors:
            confidence = 1 - var / (epsilon * epsilon)
            row.append(r'%.3f' % confidence)
        table.append_row(*row)
    saver.save_table(table, 'confidence')


if __name__ == '__main__':
    estimate_confidence()
