import functools
import os
import pickle
from weakref import WeakKeyDictionary

import config

caches = WeakKeyDictionary()
GOSSIP_OUTPUT = config.data_path(config.PAPER_DIR, 'gossip.txt')


class TildeFile:
    def __init__(self, filepath):
        self.filepath = os.path.normpath(filepath)
        self.filepath_ = self.filepath + '~'

    def __enter__(self):
        return self.filepath_

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            os.remove(self.filepath_)
        else:
            os.rename(self.filepath_, self.filepath)
            os.chmod(self.filepath, 0o444)


def ensure_dir(file_path, check=True):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if check:
        return file_path, os.path.exists(file_path)
    else:
        return file_path


def save_result_as_pickle(name=None, args_to_str=lambda *a, **kw: ''):
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            pickle_name = (func.__module__ + '.' + func.__qualname__) if name is None else name
            pickle_path = config.data_path(pickle_name + args_to_str(*args, **kwargs) + '.pkl')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                with TildeFile(pickle_path) as pickle_path:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(result, f)
                return result

        return new_func

    return decorator


def save_result_in_memory(args_to_key=lambda *a, **kw: (a, *kw.items())):
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            key = args_to_key(*args, **kwargs)
            func_cache = caches.setdefault(func, {})
            if key in func_cache:
                return func_cache[key]
            else:
                value = func_cache[key] = func(*args, **kwargs)
                return value

        return new_func

    return decorator


def save_figure(fig, name):
    fig.tight_layout(pad=0)  # Need twice
    fig.tight_layout(pad=0.05)
    save_path = os.path.join(config.PAPER_DIR, 'fig-' + name + '.pdf')
    fig.savefig(ensure_dir(save_path, check=False))
    fig.show()


def save_table(table, name):
    save_path = os.path.join(config.PAPER_DIR, 'table-' + name + '.tex')
    with open(ensure_dir(save_path, check=False), 'wt') as f:
        text = table.to_str()
        print(text)
        f.write(text)


def save_gossip(topic, *content):
    sep = '\n' + '=' * 80 + '\n\n'
    topic = topic.strip() + '\n'
    block = topic + ''.join('    %s\n' % s for s in content)
    print(block)

    file_path, existed = ensure_dir(GOSSIP_OUTPUT)
    if existed:
        with open(file_path, 'rt') as f:
            blocks = f.read().split(sep)
        for i in range(len(blocks)):
            if blocks[i].startswith(topic):
                blocks[i] = block
                break
        else:
            blocks.append(block)
    else:
        blocks = [block]
    with open(file_path, 'wt') as f:
        f.write(sep.join(blocks))
