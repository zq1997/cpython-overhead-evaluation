class LatexTable:
    def __init__(self, head_hline=True, default_column_style='c'):
        self.rows = []
        self.hlines = {1: None} if head_hline else {}
        self.default_column_style = default_column_style
        self.column_styles = {}

    def set_hline(self, row=None, rule=None):
        if row is None:
            row = len(self.rows)
        self.hlines[row] = rule

    def set_column_style(self, column, style):
        self.column_styles[column] = style

    def append_row(self, *cells):
        self.rows.append(list(cells))

    def append_column(self, *cells, style=None, row_skip=0):
        if style is not None:
            self.column_styles[max(map(len, self.rows), default=0)] = style

        if len(cells) + row_skip > len(self.rows):
            for _ in range(len(cells) + row_skip - len(self.rows)):
                self.rows.append([])
        for row, cell in zip(self.rows[row_skip:], cells):
            row.append(cell)

    def to_str(self):
        lines = []

        column_styles = [self.default_column_style] * max(map(len, self.rows))
        for c, s in self.column_styles.items():
            column_styles[c] = s
        column_styles = ''.join(column_styles)
        lines.append(r'\begin{tabular}{%s}' % column_styles)
        lines.append(r'\toprule')

        for i, row in enumerate(self.rows, 1):
            lines.append(r' & '.join(c for c in row if c is not None) + r' \\')
            if i in self.hlines:
                lines.append(self.hlines[i] or r'\midrule')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append('')

        return '\n'.join(lines)


def bracket(*cells, cmd=''):
    fmt = (r'\%s{%%s}' % cmd) if cmd else '{%s}'
    return [(fmt % c if c else '') for c in cells]


def emcell(*cells):
    return bracket(*cells, cmd='emcell')


def multi_column(n, cell, align='c'):
    return r'\multicolumn{%d}{%s}{%s}' % (n, align, cell)


def math_number(fmt, num):
    num = fmt % num
    if 'e' in num:
        base, exp = num.split('e')
        return r'$%s\times10^{%d}$' % (base, int(exp))
    else:
        return '$%s$' % num
