#!/bin/bash

# Get pip and virtualenv
sudo easy_install pip
sudo pip install --upgrade virtualenv

# Set up virtualenv
virtualenv --system-site-packages -p python3 ./tf

