#!/usr/bin/env sh
set -ex

if [ ! -d ../cpython ]; then
    git clone https://github.com/python/cpython.git ../cpython
fi

THIS_DIR="$(pwd)"

mkdir -p ../build/compile
cd ../build/compile

checkout() {
    git -C ../../cpython checkout .
    git -C ../../cpython apply "$THIS_DIR/patch/$1"
}
alias make='make -sj'

git -C ../../cpython checkout -B release v3.9.0
../../cpython/configure --enable-optimizations --without-ensurepip --prefix=
make all
make install DESTDIR=../install-release

checkout patch-gc-threshold
make install DESTDIR=../install-gc-threshold

checkout patch-count-opcodes
make install DESTDIR=../install-count-opcodes

checkout patch-count-atomic-types
make install DESTDIR=../install-count-atomic-types

checkout patch-non-threaded
make profile-removal
make all
make install DESTDIR=../install-non-threaded

git -C ../../cpython checkout .
cd "$THIS_DIR"

python3 prepare_bm.py
python3 collect_data.py
python3 estimate_confidence.py
find . -name study_\*.py -print0 | xargs -0L1 python3
