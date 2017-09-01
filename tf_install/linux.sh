#!/bin/bash

# Install pip and virtualenv
sudo apt-get install python3-pip python3-dev python-virtualenv

# Set up virtualenv
virtualenv --system-site-packages -p python3 ./tf
source ~/tf/bin/activate

# Ensure pip is installed.
# If this step fails, run
#   pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp34-cp34m-linux_x86_64.whl"
easy_install -U pip

# Otherwise, just install tensorflow
pip3 install --upgrade tensorflow

# And test
python ./test.py