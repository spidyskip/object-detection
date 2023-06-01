#!/bin/bash

# Install Python
sudo apt update
sudo apt install python3

# Install pip
sudo apt install python3-pip

# Upgrade pip
pip3 install --upgrade pip

# Install requirements
pip3 install -r requirements.txt

# Print installed Python version and installed packages
python3 --version
pip3 freeze
