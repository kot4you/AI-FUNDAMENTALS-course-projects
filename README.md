# AI Course Projects (Tasks 1–4)

## About
This repository contains my solutions for *Artificial Intelligence Fundamentals* lab assignments (Tasks *1–4*) implemented in Python.  
Each task focuses on a different AI concept: *fuzzy control*, *graph search*, *a single neuron classifier*, and a *shallow neural network*.

## Tasks included

### Task 1 — Fuzzy control (Pong)
A Pong-style game built with *pygame*, where the paddle is controlled by a *fuzzy logic controller* (Mamdani / TSK).
*Run:*
python ponggood.py

---

### Task 2 — Search algorithms (Snake route finder)
Snake game with an AI player that plans a path to food using multiple search strategies:
- *BFS* (▲)
- *DFS* (▼)
- *Dijkstra* (◄)
- *A\** (►)

*Run:*
python SNAKE_Igor_Kocik.py

---

### Task 3 — Single neuron (GUI + decision boundary)
Interactive *matplotlib GUI* demonstrating learning with a *single neuron / perceptron* and visualizing the *decision boundary* during training.

*Run:*
python one_neuron.py

---

### Task 4 — Shallow neural network (FCNN + MNIST)
A simple *fully-connected neural network (shallow MLP)* with an interactive *Streamlit* GUI.  
Supports MNIST training (downloads on first run) and shows training progress + accuracy.

*Run:*
streamlit run shallow_neural_network.py

---

## Setup

### Requirements
- Python 3.10+
- Recommended: create a virtual environment

Install dependencies:
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib pygame scikit-fuzzy scikit-learn streamlit

Note (Linux): if matplotlib GUI does not open, install Tk:
sudo apt-get install python3-tk


## Repo contents
- ponggood.py — Task 1 (Fuzzy Pong)
- SNAKE_Igor_Kocik.py — Task 2 (Snake + search)
- one_neuron.py — Task 3 (Single neuron GUI)
- shallow_neural_network.py — Task 4 (Shallow NN + MNIST Streamlit app)
