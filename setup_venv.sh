#!/bin/sh

python3 -m venv .env
source .env/bin/activate
pip3 install --upgrade pip

pip3 install -U scikit-learn

pip3 install ipykernel
pip3 install matplotlib
pip3 install networkx

pip3 install pennylane 
pip3 install pennylane-qiskit
pip3 install autograd jax jaxlib

pip3 install pandas

pip3 install qiskit
pip3 install qiskit_machine_learning

pip3 install pylatexenc

pip3 install scikit-learn
