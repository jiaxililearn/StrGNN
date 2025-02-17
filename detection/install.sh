#!/bin/bash

cd ../
cd pytorch_DGCNN
cd lib
make clean
make -j4
cd "$(dirname "$0")"
pip install --user numpy
pip install --user scipy
pip install --user networkx
pip install --user tqdm
pip install --user scikit-learn
pip install --user gensim
