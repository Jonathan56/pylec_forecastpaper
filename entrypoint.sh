#!/bin/bash

# Install Pylec
pip install -e .

# Go in study
cd studies

# Execute python file passed in parameter or CMD
exec python execute.py "$@"
