import json
import os
import pickle

import numpy as np

import config
from util import saver
from util import log
from util import git
from util import record_collect_helper
from util import eval_opcodes
from util import array_helper
from util import benchmarks

EVAL_SRC = os.path.join(config.CPY_SRC_DIR, 'Python', 'ceval.c')
EVAL_FUNC = '_PyEval_EvalFrameDefault'


class PerfData:
    def __init__(self, filepath):
        self.open_it = lambda: open(filepath, 'rb', buffering=1 << 20)
        with self.open_it() as f:
            header = f.readline()
            self.data_offset = f.tell()
            f.seek(int(header.decode().strip(), 16))
            self.bin_names, self.src_names, self.func_names = pickle.load(f)

    def iter_data(self):
        with self.open_it() as f:
            f.seek(self.data_offset)
            while True:
                data = pickle.load(f)
                if data is None:
                    return
                yield data

    def dump_decoded(self):
        for binary, addr, inlined in self.iter_data():
            print('\t0x%x @ %s' % (addr, self.bin_names[binary]))
            for src, func, line in inlined:
                print('%s:%d %s' % (self.func_names[func], line, self.src_names[src]))
            print()

    def dump_decoded_cg(self):
        for call_graph in self.iter_data():
            for binary, addr, inlined in call_graph:
                print('\t(0x%x @ %s)' % (addr, self.bin_names[binary]))
                for src, func, line in inlined:
                    print('%s:%d %s' % (self.func_names[func], line, self.src_names[src]))
            print()


def make_filter(names, targets):
    if targets is None:
        return np.full_like(names, True, np.bool)
    elif isinstance(targets, str):
        return names == targets
    else:
        final_filter = np.full_like(names, False, np.bool)
        for target in targets:
            final_filter |= names == target
        return final_filter


@saver.save_result_in_memory()
@saver.save_result_as_pickle('', lambda d: d)
def post_process(data_name):
    log.title('post-process: ' + data_name)
    bm_names = benchmarks.list_names()

    bm_tables = []
    bin_names = record_collect_helper.EncodeDict()
    src_names = record_collect_helper.EncodeDict()
    func_names = record_collect_helper.EncodeDict()

    for bm_name in bm_names:
        perf_data = PerfData(config.data_path(data_name, bm_name + '.pkl'))
        # bin(0), src(1), func(2), line(3), inline-src(4), inline-func(5), inline-line(6)
        bm_table = np.asarray([[d[0], *d[2][-1], *d[2][0]] for d in perf_data.iter_data()], dtype=np.int32)
        bin_names.update_and_remap(perf_data.bin_names, bm_table[:, 0])
        src_names.update_and_remap(perf_data.src_names, bm_table[:, 1], bm_table[:, 4])
        func_names.update_and_remap(perf_data.func_names, bm_table[:, 2], bm_table[:, 5])
        bm_tables.append(bm_table)
        log.item(bm_name, len(bm_table))

    return (
        bm_tables,
        np.asarray(bin_names.keys()), np.asarray(src_names.keys()), np.asarray(func_names.keys())
    )


def load_name_dist(
        what=None, data_name=config.D_RECORD_CYCLES,
        inline=False, bin_name=config.CPY_EXE):
    what = ('bin', 'src', 'func').index(what)
    bm_tables, bin_names, src_names, func_names = post_process(data_name)
    names = (bin_names, src_names, func_names)[what]
    length = len(names)

    if what == 0:
        dist = np.vstack([np.bincount(t[:, 0], minlength=length) for t in bm_tables])
    else:
        dist = np.empty([len(bm_tables), length], np.int)
        index = [[1, 2], [4, 5]][inline][what - 1]
        bin_filter = make_filter(bin_names, bin_name)

        for row, table in zip(dist, bm_tables):
            row[:] = np.bincount(table[bin_filter[table[:, 0]], index], minlength=dist.shape[1])

    names.flags.writeable = False
    return dist, names


def load_line_dist(
        data_name=config.D_RECORD_CYCLES,
        bin_name=config.CPY_EXE,
        src_name=EVAL_SRC,
        func_name=EVAL_FUNC,
        inline=False, max_line=0):
    if max_line == 0:
        max_line = git.count_text_line(git.read_file_by_full_path(src_name))

    bm_tables, bin_names, src_names, func_names = post_process(data_name)
    bin_filter = make_filter(bin_names, bin_name)
    src_filter = make_filter(src_names, src_name)
    func_filter = make_filter(func_names, func_name)
    inline_offset = 3 if inline else 0

    dist = np.zeros([len(bm_tables), max_line + 1], np.int)
    for row, table in zip(dist, bm_tables):
        selector = bin_filter[table[:, 0]]
        selector &= src_filter[table[:, 1 + inline_offset]]
        selector &= func_filter[table[:, 2 + inline_offset]]
        if max_line == 1:
            row[1] = selector.sum()
            row[0] = len(table) - row[1]
        else:
            count = np.bincount(table[selector, 3 + inline_offset])
            row[:len(count)] = count
            row[0] += len(table) - count.sum()
    return dist


@saver.save_result_in_memory()
@saver.save_result_as_pickle('', lambda d: d)
def load_line_dist_cg(data_name):
    log.title('post-process (with call graph): ' + data_name)

    with open(config.CPY_FUNC_JSON) as f:
        native_functions = set(map(tuple, json.load(f)))
    obj_src_files = [os.path.join(config.CPY_SRC_DIR, 'Objects', s) for s in (
        'longobject.c',
        'floatobject.c',
        'boolobject.c',
        'complexobject.c',
        'unicodeobject.c',
        'bytesobject.c',
        'tupleobject.c',
        'listobject.c',
        'setobject.c',
        'dictobject.c',

        'call.c'
    )]

    bm_names = benchmarks.list_names()

    line_number = git.count_text_line(git.read_file_by_full_path(EVAL_SRC))
    dist = np.zeros([len(bm_names), 4, line_number + 1], np.int)
    is_setup_or_cleanup = eval_opcodes.eval_frame_setup_or_cleanup()

    for bm_dist, bm_name in zip(dist, bm_names):
        perf_data = PerfData(config.data_path(data_name, bm_name + '.pkl'))

        cpy_exe = perf_data.bin_names.index(config.CPY_EXE)
        eval_src_func = (
            perf_data.src_names.index(EVAL_SRC),
            perf_data.func_names.index(EVAL_FUNC)
        )
        entry_src_func = (
            perf_data.src_names.index(os.path.join(config.CPY_SRC_DIR, 'Modules', 'main.c')),
            perf_data.func_names.index('Py_BytesMain')
        )
        obj_src = {i for i, s in enumerate(perf_data.src_names) if s in obj_src_files}

        # When sampling, we can only dump a small length of the stack,
        # so sometimes the call graph information is insufficient.
        # Fortunately, this does not happen very often.
        insufficient_cg = 0
        for call_graph in perf_data.iter_data():
            native_func = 0
            obj_func = 0
            on_top = 1

            for binary, _, inlined in call_graph:
                src, func, line = inlined[-1]
                if binary == cpy_exe:
                    src_func = (src, func)
                    if src_func == eval_src_func:
                        if not is_setup_or_cleanup[line]:
                            bm_dist[:, line] += [1, native_func, obj_func, on_top]
                            break
                    elif src_func == entry_src_func:
                        bm_dist[:, 0] += [1, native_func, obj_func, on_top]
                        break
                    else:
                        src_name = perf_data.src_names[src]
                        func_name = perf_data.func_names[func]
                        src_func_name = (src_name, func_name)
                        if src_func_name in native_functions:
                            native_func = 1
                        if src in obj_src:
                            obj_func = 1
                elif not perf_data.bin_names[binary].startswith('['):
                    native_func = 1
                    obj_func = 1
                on_top = 0
            else:
                bm_dist[:, 0] += [1, native_func, obj_func, on_top]
                insufficient_cg += 1

        total = bm_dist.sum()
        log.item(bm_name, '%d/%d=%.4f' % (insufficient_cg, total, insufficient_cg / total))

    return dist


def load_opcode_dist_cg(data_name=config.D_RECORD_CYCLES_CG):
    dist = load_line_dist_cg(data_name)
    opcode_remap_arr, opcode_names = eval_opcodes.line_to_opcode()
    is_non_opcode = opcode_remap_arr == -1
    category_remap_arr, category_names = eval_opcodes.opcode_to_category()
    is_call_opcode = category_remap_arr == list(category_names).index('function call')

    # for <NON_OPCODE>
    non_opcode = dist[:, 0, is_non_opcode].sum(-1)[..., None]
    dist[:, :, is_non_opcode] = 0
    dist = array_helper.remap_dist_by_arr(dist, opcode_remap_arr, len(opcode_names))

    # for <NATIVE_CALLS>
    dist[:, 0, is_call_opcode] -= dist[:, 1, is_call_opcode]
    native_calls = dist[:, 1, is_call_opcode].sum(-1)[..., None]

    dist = np.concatenate([dist[:, 0, :], non_opcode, native_calls], axis=-1)
    opcode_names = [*opcode_names, '<NON_OPCODE>', '<NATIVE_CALLS>']
    return dist, np.asarray(opcode_names)
