# AI Course Projects (Tasks 4.1–4.4)

## About
This repository contains my solutions for *Artificial Intelligence Fundamentals* lab assignments (Tasks *4.1–4.4*) implemented in Python.  
Each task focuses on a different AI concept: *fuzzy control*, *graph search*, *a single neuron classifier*, and a *shallow neural network*.

## Tasks included

### Task 4.1 — Fuzzy control (Pong)
A Pong-style game built with *pygame*, where the paddle is controlled by a *fuzzy logic controller* (scikit-fuzzy).  
*Focus:* fuzzy sets, rules, defuzzification, control systems.

Run:
python Task1_Szymon_Sobiech.py

---

### Task 4.2 — Search algorithms (Snake route finder)
A Snake game with an AI player that can use different search algorithms to find a path to food:
- *BFS* (press ▲)
- *DFS* (press ▼)
- *Dijkstra* (press ◄)
- *A\** (press ►)

*Focus:* state space search, graph traversal, heuristics, shortest path.

Run:
python Task2_Szymon_Sobiech.py

---

### Task 4.3 — Single neuron (GUI + decision boundary)
An interactive *matplotlib GUI* that demonstrates learning with a *single neuron / perceptron* and visualizes the *decision boundary* during training.  
*Focus:* linear classification, perceptron learning rule, decision boundary visualization.

Run:
python Task3_Szymon_Sobiech.py

---

### Task 4.4 — Shallow neural network (FCNN + MNIST)
A simple *fully-connected neural network (shallow MLP)* implemented in NumPy with an interactive GUI (matplotlib widgets).  
Trains and evaluates on *MNIST*, showing training progress and accuracy.  
*Focus:* backpropagation, activation functions, training loop, classification metrics.

Run:
python Task4_Szymon_Sobiech.py

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
python -m pip install pygame scikit-fuzzy numpy matplotlib

Note (Linux): if matplotlib GUI does not open, install Tk:
sudo apt-get install python3-tk


## Controls (where applicable)
- *Task 4.2 (Snake):* Arrow keys select the search algorithm (▲ BFS / ▼ DFS / ◄ Dijkstra / ► A*)
- *Task 4.1 (Pong):* gameplay is autonomous via fuzzy control (no manual paddle control)
- *Tasks 4.3–4.4:* use GUI buttons/inputs in the matplotlib window

## Repo contents
- Task1_Szymon_Sobiech.py — Task 4.1 (Fuzzy Pong)
- Task2_Szymon_Sobiech.py — Task 4.2 (Snake + search)
- Task3_Szymon_Sobiech.py — Task 4.3 (Single neuron GUI)
- Task4_Szymon_Sobiech.py — Task 4.4 (Shallow NN + MNIST GUI)
