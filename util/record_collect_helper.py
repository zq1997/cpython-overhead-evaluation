import multiprocessing
import subprocess
import os
import re
import pickle

import numpy as np

import config
from util import benchmarks
from util import log
from util import saver

FAKE_ADDR2LINE_PATH = config.data_path('fake_addr2line')


class EncodeDict(dict):
    def __missing__(self, key):
        ret = self[key] = len(self)
        return ret

    def keys(self):
        return [kv[0] for kv in sorted(super().items(), key=lambda kv: kv[1])]

    def update_and_remap(self, name_list, *np_arrays):
        remap_arr = np.empty(len(name_list), dtype=np.int)
        for v in np.unique([y for x in map(np.unique, np_arrays) for y in x]):
            remap_arr[v] = self[name_list[v]]
        for arr in np_arrays:
            arr[:] = remap_arr[arr]


class PerfDataWriter:
    def __init__(self, filepath):
        self.event_names = EncodeDict()
        self.bin_names = EncodeDict()
        self.src_names = EncodeDict()
        self.func_names = EncodeDict()
        self.f = open(filepath, 'wb', buffering=1 << 20)
        self.f.write(('%16X\n' % 0).encode())

    def close(self):
        pickle.dump(None, self.f)
        offset = self.f.tell()
        assert offset <= (1 << 64 - 1)
        pickle.dump((
            self.bin_names.keys(),
            self.src_names.keys(),
            self.func_names.keys(),
        ), self.f)
        self.f.seek(0)
        self.f.write(('%16X\n' % offset).encode())
        self.f.close()


class Addr2Line:
    pattern = re.compile(r'(.*):(\d+|\?)(?: \(discriminator \d+\))?\n')

    def __init__(self, dso):
        self.proc = None
        if os.path.isfile(dso):
            self.proc = subprocess.Popen(
                ['addr2line', '-afie', dso],
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            assert self.proc.wait() == 0

    def get(self, addr):
        if self.proc is None:
            return [('??', '??', 0)]
        print('%x\n0' % addr, file=self.proc.stdin, flush=True)
        self.proc.stdout.readline()
        inlined = []
        for func_name in self.proc.stdout:
            if func_name[0] == '0':
                self.proc.stdout.readline()
                self.proc.stdout.readline()
                return inlined
            else:
                src_name, line = re.fullmatch(Addr2Line.pattern, self.proc.stdout.readline()).groups()
                src_name = os.path.normpath(src_name) if src_name else '??'
                func_name = func_name.rstrip()
                line = 0 if line == '?' else int(line)
                inlined.append((src_name, func_name, line))


def perf_export(raw_files, pkl_file, call_graph, comm):
    addr2line_dict = {}
    writer = PerfDataWriter(pkl_file)
    unknown_result = (
        writer.bin_names['[unknown]'], 0,
        [(writer.src_names['??'], writer.func_names['??'], 0)]
    )
    dso_pattern = re.compile(r'\s*\S+? \((.+)\)\n')
    addr_pattern = re.compile(r'\s*\S+?\[(.+)\]\n')

    def get_src_line(dso_input, stdout):
        dso = re.fullmatch(dso_pattern, dso_input).group(1)
        if dso == '[unknown]':
            append_func(unknown_result)
        else:
            addr = int(re.fullmatch(addr_pattern, stdout.readline()).group(1), 16)
            addr2line = addr2line_dict.get(dso, None)
            if addr2line is None:
                addr2line = addr2line_dict[dso] = Addr2Line(dso)
            inlined = addr2line.get(addr)
            inlined = [(writer.src_names[s], writer.func_names[f], l) for s, f, l in inlined]
            append_func((writer.bin_names[dso], addr, inlined))

    for raw_file in raw_files:
        proc = subprocess.Popen(
            [
                'perf', 'script',
                '-Fip,dso,srcline', '--full-source-path', '--no-inline',
                '-c', comm, '-i', raw_file
            ],
            env={'PATH': '%s:%s' % (FAKE_ADDR2LINE_PATH, os.environ['PATH'])},
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        if call_graph:
            call_graph_data = []
            append_func = call_graph_data.append
            while True:
                dso_input = proc.stdout.readline()
                if not dso_input:
                    break
                if dso_input.isspace():
                    if call_graph_data:
                        pickle.dump(call_graph_data, writer.f)
                    call_graph_data.clear()
                    continue
                get_src_line(dso_input, proc.stdout)
        else:
            append_func = lambda x: pickle.dump(x, writer.f)
            while True:
                dso_input = proc.stdout.readline()
                if not dso_input:
                    break
                get_src_line(dso_input, proc.stdout)
        assert proc.wait() == 0

    writer.close()
    for addr2line in addr2line_dict.values():
        addr2line.close()


def _do_record(args):
    file_prefix, cpython, perf_args, bm_meta = args
    pkl_file = file_prefix + '.pkl'
    if not os.path.exists(pkl_file):
        logger = log.TimeLogger(bm_meta[0])
        with logger.add_message('warm'):
            warm_file = file_prefix + '.warming'
            for _ in range(3):
                perf_args[-2] = warm_file
                benchmarks.run_bm(bm_meta, cpython, config.WARM_UP_DURATION, pre_args=perf_args, stderr=False)
                os.remove(warm_file)
        with logger.add_message('record'):
            for i in range(config.RECORD_REPEAT):
                with saver.TildeFile(file_prefix + ('-%d.raw' % i)) as raw_file_:
                    perf_args[-2] = raw_file_
                    benchmarks.run_bm(bm_meta, cpython, pre_args=perf_args, stderr=False)


def _do_export(args):
    file_prefix, call_graph, bm_meta = args
    raw_files = [file_prefix + ('-%d.raw' % i) for i in range(config.RECORD_REPEAT)]
    pkl_file = file_prefix + '.pkl'
    if not os.path.exists(pkl_file):
        with log.TimeLogger(bm_meta[0]).add_message('export'):
            with saver.TildeFile(pkl_file) as pkl_file_:
                perf_export(raw_files, pkl_file_, call_graph, os.path.basename(config.CPY_EXE))


def collect(data_name, event, period, cpython=config.CPY_EXE, call_graph=0):
    perf_args = [x for x in (
        'perf', 'record', '-B',
        '-e', event, '-c', str(period),
        ('--call-graph=dwarf,%d' % call_graph) if call_graph else None,
        '-o', '',
        '--'
    ) if x is not None]

    file_prefix = saver.ensure_dir(config.data_path(data_name, ''), check=False)
    record_args_list = []
    export_args_list = []
    for bm_meta in benchmarks.list_meta():
        bm_file_prefix = file_prefix + bm_meta[0]
        record_args_list.append([bm_file_prefix, cpython, list(perf_args), bm_meta])
        export_args_list.append([bm_file_prefix, call_graph != 0, bm_meta])

    with multiprocessing.Pool(config.PARALLEL) as pool:
        log.title('perf record: ' + data_name)
        pool.map(_do_record, record_args_list)

    if not os.path.exists(FAKE_ADDR2LINE_PATH):
        os.makedirs(FAKE_ADDR2LINE_PATH, exist_ok=True)
        os.symlink('/bin/false', os.path.join(FAKE_ADDR2LINE_PATH, 'addr2line'))

    with multiprocessing.Pool() as pool:
        log.title('export: ' + data_name)
        pool.map(_do_export, export_args_list)
