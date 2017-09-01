#!/bin/bash

# Get pip and virtualenv
sudo easy_install pip
sudo pip install --upgrade virtualenv

# Set up virtualenv
virtualenv --system-site-packages -p python3 ./tf

# Activate virtualenv
source ~/tf/bin/activate

# Ensure pip is installed. If this step fails, run
#  pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
easy_install -U pip

# Otherwise, just install tensorflow
pip3 install --upgrade tensorflow

python ./test.py