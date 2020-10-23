#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"

# creating local venv
. $HERE/../shared/setup.sh $HERE

TARGET_DIR="${HERE}/lib"

rm -Rf ${TARGET_DIR}
mkdir ${HERE}/lib

# sudo apt install graphviz swig libgraphviz-dev
# ln -s ~/phd/code/pynisher ${TARGET_DIR}/pynisher
# ln -s ~/phd/code/sklearn-components ${TARGET_DIR}/sklearn-components
# ln -s ~/phd/code/dswizard ${TARGET_DIR}/dswizard

git clone --depth 1 --single-branch --branch master --recurse-submodules git@github.com:Ennosigaeon/pynisher.git ${TARGET_DIR}/pynisher
git clone --depth 1 --single-branch --branch master --recurse-submodules git@github.com:Ennosigaeon/sklearn-components.git ${TARGET_DIR}/sklearn-components
git clone --depth 1 --single-branch --branch master --recurse-submodules git@github.com:Ennosigaeon/dswizzard.git ${TARGET_DIR}/dswizard

PIP install -e ${TARGET_DIR}/pynisher
PIP install -e ${TARGET_DIR}/sklearn-components
PIP install -e ${TARGET_DIR}/dswizard

