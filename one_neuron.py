from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

# -----------------------------
# 1. Activation Functions & Derivatives
# -----------------------------

# --- Heaviside ---
def heaviside(s: np.ndarray) -> np.ndarray:
    return (s >= 0).astype(float)

def d_heaviside(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.ones_like(s)

# --- Logistic (Sigmoid) ---
def sigmoid(s: np.ndarray, beta: float = 1.0) -> np.ndarray:
    s = np.clip(s, -60, 60)
    return 1.0 / (1.0 + np.exp(-beta * s))

def d_sigmoid(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return beta * y * (1.0 - y)

# --- Tanh ---
def tanh_act(s: np.ndarray) -> np.ndarray:
    return np.tanh(s)

def d_tanh(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return 1.0 - y**2

# --- Sin ---
def sin_act(s: np.ndarray) -> np.ndarray:
    return np.sin(s)

def d_sin(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.cos(s)

# --- Sign ---
def sign_act(s: np.ndarray) -> np.ndarray:
    return np.sign(s)

def d_sign(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.ones_like(s)

# --- ReLU ---
def relu(s: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, s)

def d_relu(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return (s > 0).astype(float)

# --- Leaky ReLU ---
def leaky_relu(s: np.ndarray) -> np.ndarray:
    return np.where(s > 0, s, 0.01 * s)

def d_leaky_relu(s: np.ndarray, y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.where(s > 0, 1.0, 0.01)

# Dictionary
# "threshold@0.5" -> standard for 0..1 output
# "threshold@0"   -> standard for -1..1 or 0..inf output
ACTS = {
    "Heaviside": (heaviside, d_heaviside, "threshold@0.5"),
    "Logistic":  (sigmoid, d_sigmoid, "threshold@0.5"),
    "tanh":      (tanh_act, d_tanh, "threshold@0"),
    "sin":       (sin_act, d_sin, "threshold@0"),
    "sign":      (sign_act, d_sign, "threshold@0"),
    "ReLU":      (relu, d_relu, "threshold@0"),
    "Leaky ReLU":(leaky_relu, d_leaky_relu, "threshold@0"),
}


# -----------------------------
# 2. Data Generation
# -----------------------------
@dataclass
class GMMParams:
    means: np.ndarray
    variances: np.ndarray

def generate_class_gmm(rng, modes, spm, mu_min, mu_max, var_min, var_max):
    means = rng.uniform(mu_min, mu_max, size=(modes, 2))
    variances = rng.uniform(var_min, var_max, size=(modes, 2))
    parts = []
    for k in range(modes):
        cov = np.diag(variances[k])
        Xk = rng.multivariate_normal(means[k], cov, size=spm)
        parts.append(Xk)
    X = np.vstack(parts) if parts else np.empty((0, 2))
    return X, GMMParams(means=means, variances=variances)

def generate_dataset(seed, modes, spm, mu_min, mu_max, var_min, var_max):
    rng = np.random.default_rng(seed)
    X0, p0 = generate_class_gmm(rng, modes, spm, mu_min, mu_max, var_min, var_max)
    X1, p1 = generate_class_gmm(rng, modes, spm, mu_min, mu_max, var_min, var_max)
    
    y0 = np.zeros(len(X0))
    y1 = np.ones(len(X1))
    
    X = np.vstack([X0, X1])
    d = np.concatenate([y0, y1])
    
    idx = rng.permutation(len(d))
    return X[idx], d[idx], {0: p0, 1: p1}


# -----------------------------
# 3. Neuron Class
# -----------------------------
class Neuron:
    def __init__(self, seed: int = 0, activation: str = "Heaviside", beta: float = 1.0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.6, size=(3,)) 
        self.activation = activation
        self.beta = beta

    def _augment(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones((X.shape[0],)), X])

    def net(self, X: np.ndarray) -> np.ndarray:
        Xa = self._augment(X)
        return Xa @ self.w

    def forward(self, X: np.ndarray) -> np.ndarray:
        act, _, _ = ACTS[self.activation]
        s = self.net(X)
        if self.activation == "Logistic":
            return act(s, beta=self.beta)
        return act(s)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        y = self.forward(X)
        _, _, rule = ACTS[self.activation]
        
        if rule == "threshold@0.5":
            return (y >= 0.5).astype(float)
        return (y > 0.0).astype(float)

    def accuracy(self, X: np.ndarray, d: np.ndarray) -> float:
        preds = self.predict_class(X)
        return float(np.mean(preds == d))

    def train_epoch(self, X: np.ndarray, d: np.ndarray, eta: float, shuffle: bool = True, seed: int | None = None) -> float:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(d)) if shuffle else np.arange(len(d))

        Xa = self._augment(X)
        act, dact, _ = ACTS[self.activation]

        loss = 0.0
        for i in idx:
            x = Xa[i]
            dj = float(d[i])
            s = float(self.w @ x)
            
            if self.activation == "Logistic":
                y = float(act(np.array([s]), beta=self.beta)[0])
                fp = float(dact(np.array([s]), np.array([y]), beta=self.beta)[0])
            else:
                y = float(act(np.array([s]))[0])
                fp = float(dact(np.array([s]), np.array([y]), beta=self.beta)[0])

            loss += 0.5 * (dj - y) ** 2

            # Eq (1)
            self.w += eta * (dj - y) * fp * x

        return loss / max(1, len(d))


# -----------------------------
# 4. GUI & Main
# -----------------------------
def decision_grid(neuron: Neuron, X: np.ndarray, pad: float = 0.6, n: int = 200):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    Z = neuron.predict_class(grid).reshape(YY.shape)
    return XX, YY, Z

def draw_scene(ax, X, d, neuron: Neuron, show_means: bool, params: Dict[int, GMMParams], grid_n: int):
    ax.clear()
    
    # Boundary
    XX, YY, Z = decision_grid(neuron, X, n=grid_n)
    ax.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5], alpha=0.25, colors=["#ff6b6b", "#4dabf7"])
    ax.contour(XX, YY, Z, levels=[0.5], colors=["#222"], linewidths=1.2, alpha=0.9)

    # Points
    mask1 = (d == 1)
    ax.scatter(X[~mask1, 0], X[~mask1, 1], s=26, c="#c92a2a", label="Class 0", edgecolors="white", linewidths=0.3)
    ax.scatter(X[mask1, 0], X[mask1, 1], s=26, c="#1864ab", label="Class 1", edgecolors="white", linewidths=0.3)

    # Means
    if show_means and params is not None:
        ax.scatter(params[0].means[:, 0], params[0].means[:, 1], marker="x", s=90, c="#5f3dc4", linewidths=2, label="Means c0")
        ax.scatter(params[1].means[:, 0], params[1].means[:, 1], marker="x", s=90, c="#2f9e44", linewidths=2, label="Means c1")

    ax.set_title(f"Act: {neuron.activation} | Acc: {neuron.accuracy(X, d)*100:.1f}% | Beta: {neuron.beta}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", frameon=True, fontsize='small')

def main():
    state = {
        "seed": 42, "modes": 2, "spm": 60,
        "mu_min": -1.5, "mu_max": 1.5, "var_min": 0.05, "var_max": 0.5,
        "eta": 0.01, "epochs": 20, "beta": 1.0,
        "activation": "Heaviside",
        "show_means": True, "grid_n": 150, "shuffle": True
    }

    X, d, params = generate_dataset(state["seed"], state["modes"], state["spm"], 
                                    state["mu_min"], state["mu_max"], 
                                    state["var_min"], state["var_max"])
    neuron = Neuron(seed=state["seed"]+99, activation=state["activation"], beta=state["beta"])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0.35, 0.10, 0.60, 0.85]) 

    ui_x, ui_w, ui_h = 0.05, 0.22, 0.03
    
    s_seed   = Slider(fig.add_axes([ui_x, 0.92, ui_w, ui_h]), "Seed", 0, 1000, valinit=state["seed"], valstep=1)
    s_eta    = Slider(fig.add_axes([ui_x, 0.88, ui_w, ui_h]), "Eta", 0.0001, 1.0, valinit=state["eta"], valstep=0.001)
    s_epochs = Slider(fig.add_axes([ui_x, 0.84, ui_w, ui_h]), "Epochs", 1, 200, valinit=state["epochs"], valstep=1)
    s_beta   = Slider(fig.add_axes([ui_x, 0.80, ui_w, ui_h]), "Beta", 0.1, 10.0, valinit=state["beta"], valstep=0.1)
    s_modes  = Slider(fig.add_axes([ui_x, 0.74, ui_w, ui_h]), "Modes", 1, 5, valinit=state["modes"], valstep=1)
    s_spm    = Slider(fig.add_axes([ui_x, 0.70, ui_w, ui_h]), "N/Mode", 10, 200, valinit=state["spm"], valstep=10)
    s_var    = Slider(fig.add_axes([ui_x, 0.66, ui_w, ui_h]), "Var Max", 0.1, 2.0, valinit=state["var_max"])

    ax_radio = fig.add_axes([ui_x, 0.35, ui_w, 0.25])
    ax_radio.set_title("Activation", loc='left', fontsize=9)
    radio = RadioButtons(ax_radio, list(ACTS.keys()), active=list(ACTS.keys()).index(state["activation"]))

    b_gen   = Button(fig.add_axes([ui_x, 0.25, 0.10, 0.05]), "Generate")
    b_reset = Button(fig.add_axes([0.17, 0.25, 0.10, 0.05]), "Reset W")
    b_train = Button(fig.add_axes([ui_x, 0.18, ui_w, 0.06]), "TRAIN", color="#4dabf7", hovercolor="#339af0")
    check   = CheckButtons(fig.add_axes([ui_x, 0.10, ui_w, 0.07]), ["Show Means"], [state["show_means"]])

    def update_params():
        state["seed"] = int(s_seed.val)
        state["eta"] = s_eta.val
        state["epochs"] = int(s_epochs.val)
        state["beta"] = s_beta.val
        state["modes"] = int(s_modes.val)
        state["spm"] = int(s_spm.val)
        state["var_max"] = s_var.val
        state["activation"] = radio.value_selected
        state["show_means"] = check.get_status()[0]

    def on_gen(event):
        new_seed = np.random.randint(0, 1000)
        s_seed.eventson = False 
        s_seed.set_val(new_seed)
        s_seed.eventson = True
        
        update_params()
        nonlocal X, d, params
        X, d, params = generate_dataset(state["seed"], state["modes"], state["spm"], 
                                        state["mu_min"], state["mu_max"], 
                                        state["var_min"], state["var_max"])
        on_reset(None)

    def on_reset(event):
        update_params()
        nonlocal neuron
        neuron = Neuron(seed=state["seed"] + 123, activation=state["activation"], beta=state["beta"])
        draw_scene(ax, X, d, neuron, state["show_means"], params, state["grid_n"])
        fig.canvas.draw_idle()

    def on_train(event):
        update_params()
        neuron.activation = state["activation"]
        neuron.beta = state["beta"]
        for _ in range(state["epochs"]):
            neuron.train_epoch(X, d, eta=state["eta"], shuffle=state["shuffle"])
        draw_scene(ax, X, d, neuron, state["show_means"], params, state["grid_n"])
        fig.canvas.draw_idle()

    def on_ui_change(val):
        update_params()
        neuron.activation = state["activation"]
        neuron.beta = state["beta"]
        draw_scene(ax, X, d, neuron, state["show_means"], params, state["grid_n"])
        fig.canvas.draw_idle()

    b_gen.on_clicked(on_gen)
    b_reset.on_clicked(on_reset)
    b_train.on_clicked(on_train)
    radio.on_clicked(on_ui_change)
    check.on_clicked(on_ui_change)
    s_beta.on_changed(on_ui_change)

    draw_scene(ax, X, d, neuron, state["show_means"], params, state["grid_n"])
    plt.show()

if __name__ == "__main__":
    main()