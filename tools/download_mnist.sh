#!/bin/bash

DATA_DIR=mnist_data
URL_BASE=https://raw.githubusercontent.com/fgnt/mnist/master

mkdir -p $DATA_DIR

wget -O $DATA_DIR/train-images-idx3-ubyte.gz ${URL_BASE}/train-images-idx3-ubyte.gz
wget -O $DATA_DIR/train-labels-idx1-ubyte.gz ${URL_BASE}/train-labels-idx1-ubyte.gz
wget -O $DATA_DIR/t10k-images-idx3-ubyte.gz ${URL_BASE}/t10k-images-idx3-ubyte.gz
wget -O $DATA_DIR/t10k-labels-idx1-ubyte.gz ${URL_BASE}/t10k-labels-idx1-ubyte.gz

gunzip $DATA_DIR/*.gz
