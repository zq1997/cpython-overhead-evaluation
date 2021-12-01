import subprocess
import multiprocessing
import os
import types
import sys
import json

import config
from util import saver
from util import log

if not saver.ensure_dir(config.PIP_DIR)[1]:
    def pip_install(*args, python_path=config.PIP_DIR):
        subprocess.check_call(
            [config.CPY_EXE, '-m', 'pip', 'install', '--no-cache-dir', '-U', '-t', config.PIP_DIR, *args],
            env={
                'PATH': os.environ['PATH'],
                'PYTHONPATH': python_path
            }
        )

    bootstrap_dir = os.path.join(config.CPY_SRC_DIR, 'Lib', 'ensurepip', '_bundled')
    bootstrap_whl = filter(lambda f: f.endswith('.whl'), os.listdir(bootstrap_dir))
    python_path = os.path.pathsep.join(os.path.join(bootstrap_dir, f) for f in bootstrap_whl)
    pip_install('pip', python_path=python_path)
    pip_install('setuptools', 'wheel')
    pip_install('--no-deps', 'pyperformance==' + config.PYPERFORMANCE_VERSION)
    pip_install('-r', os.path.join(config.PIP_DIR, 'pyperformance', 'requirements.txt'))

if not saver.ensure_dir(config.CALIBRATE_JSON)[1]:
    benchmark_prefix = os.path.join(config.PIP_DIR, 'pyperformance', 'benchmarks', '')
    fake_run_mod = sys.modules['pyperformance.run'] = types.ModuleType('pyperformance.run')
    fake_run_mod.run_perf_script = lambda python, options, name, extra_args=(): \
        ((benchmark_prefix + 'bm_%s.py' % name), *extra_args)

    benchmark_mod = {}
    with open(benchmark_prefix + '__init__.py') as f:
        exec(f.read(), benchmark_mod)
    bm_funcs, bm_groups = benchmark_mod['get_benchmarks']()

    def calibrate(bm_name):
        scr_args = bm_funcs[bm_name](None, None)
        raw_json = saver.ensure_dir(os.path.join(config.CALIBRATE_DIR, bm_name + '.json'), check=False)
        subprocess.check_call(
            [
                config.CPY_EXE, *scr_args,
                '--inherit-environ=PYTHONPATH',
                '--worker', '-w0', '-n1', '--min-time', '%.3f' % config.WARM_UP_DURATION,
                '--calibrate-loops',
                '-o', raw_json
            ],
            stdout=subprocess.DEVNULL,
            env={'PYTHONPATH': config.PIP_DIR}
        )
        with open(raw_json, 'r') as f:
            result = json.load(f)

        bm_durations = 0
        inner_loops = result['metadata'].get('inner_loops', 1)
        for sub_result in result['benchmarks']:
            sub_inner_loops = sub_result.get('metadata', {}).get('inner_loops', inner_loops)
            bm_durations += sub_result['runs'][0]['warmups'][-1][-1] * sub_inner_loops

        log.item(bm_name, '%.3g ms' % (bm_durations * 1000))
        return [bm_name, bm_durations, scr_args]


    with multiprocessing.Pool(config.PARALLEL) as pool:
        log.title('calibrate')
        calibrated_result = pool.map(calibrate, sorted(bm_groups['all']))
    with saver.TildeFile(saver.ensure_dir(config.CALIBRATE_JSON, check=False)) as output_json:
        with open(output_json, 'wt') as f:
            json.dump(calibrated_result, f, indent=2)
