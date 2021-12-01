import subprocess
import os

import config


def read_file(filepath, version=config.CPY_VERSION):
    proc = subprocess.Popen(
        [
            'git',
            '-C', config.CPY_SRC_DIR,
            'show', version + ':' + filepath
        ],
        universal_newlines=True,
        stdout=subprocess.PIPE
    )
    content = proc.stdout.read()
    assert proc.wait() == 0
    return content


def read_file_by_full_path(full_path, version=config.CPY_VERSION):
    repo_prefix = config.CPY_SRC_DIR + os.sep
    if full_path.startswith(repo_prefix):
        return read_file(full_path[len(repo_prefix):], version)
    else:
        with open(full_path, 'rt') as f:
            return f.read()


def count_text_line(text):
    return text.count('\n') + (not text.endswith('\n') if text else 0)
