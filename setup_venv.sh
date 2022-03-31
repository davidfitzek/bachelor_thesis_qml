#!/bin/sh

python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip

pip install -U scikit-learn

pip install ipykernel
pip install matplotlib
pip install networkx

pip install pennylane 
pip install pennylane-qiskit
pip install autograd jax jaxlib