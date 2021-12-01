# Evaluate the overhead of CPython with Linux Perf



## System Requirements

The system should be Linux

## Dependencies

Install and enable Linux perf. If successful, the following command will give meaningful output.

```sh
perf stat -- sleep 1
```

Install the packages required to compile CPython. For ubuntu, use the following command

```sh
apt-get -yq install \
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    libgdbm-dev \
    tk-dev \
    lzma \
    lzma-dev \
    liblzma-dev \
    libffi-dev \
    uuid-dev \
    xvfb
```

Install some python packages.

```sh
python3 -m pip install matplotlib numpy scipy pyelftools Pygments psutil
```

## Run

Create a empty directory and clone this repository as its subdirectory. The `run.sh` script will reproduce the entire experiment.

``` sh
mkdir cpython-overhead
git clone https://github.com/zq1997/cpython-overhead-evaluation.git evaluation
cd evaluation
./run.sh
```

He will generate some other directories under `cpython-overhead`:

- `cpython`, the CPython source tree
- `build`,  the compilation and installation directory
- `benchmark`, for pyperformance benchmark suite
- `paper`, the PDF figure and tex tables for the paper