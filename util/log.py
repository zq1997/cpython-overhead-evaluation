import time


def title(title):
    print('=' * 20, title, '=' * 20, flush=True)


def item(bm_name, *info):
    print(bm_name + ':', *info, flush=True)


def message(*msgs):
    print(*msgs, flush=True, end='')


class TimeLogger:
    def __init__(self, message):
        self.message = message + ':'

    def add_message(self, message):
        self.message += '\t%s' % message
        return self

    def __del__(self):
        if self.message is not None:
            print(self.message, flush=True)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        error = exc_type is not None
        self.message += '[%.1fs%s]' % (time.time() - self.start_time, ',ERROR' if error else '')
        if error:
            print(self.message, flush=True)
            self.message = None
