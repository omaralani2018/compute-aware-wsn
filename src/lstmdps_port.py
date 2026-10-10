"""
Port of LSTM-DPS (Fathalla et al. [18]) from the authors' released code
(github.com/Ahmed-Fathalla/LSTM-DPS-for-IoT, utils/_run.py run_exp and
utils/_experiment.py), restricted to the configuration they used on Intel:
Apply_diff = True, LMS_w = LSTM_w = 3, initialization_period = 3,
LMS_period = 20, employ_update_policy = False (read from every saved Intel
experiment). The LSTM forward pass is NumPy, Keras 2.2.4 semantics: gate order
i, f, c, o; recurrent activation hard_sigmoid; state activation tanh.
"""
import numpy as np, h5py, json

LMS_W = np.array([0.88744788, -0.28791624, 0.52442267])   # padasip FilterLMS(3, mu=.01, w=...)
# The released loop only ever calls filt.predict(), never filt.adapt(), so these
# weights stay fixed: the "LMS" phase is a fixed 3-tap linear predictor of differences.

def _act(name):
    return {"linear": lambda z: z, "relu": lambda z: np.maximum(z, 0.0),
            "tanh": np.tanh, "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
            "hard_sigmoid": lambda z: np.clip(0.2 * z + 0.5, 0.0, 1.0)}[name]


class NumpyLSTMDPSModel:
    """Input (3,1) -> LSTM(u) -> Dense(d1, act1) [-> Dense(d2, act2)] -> Dense(1)."""
    def __init__(self, layers):
        self.layers = layers                              # list of dicts
    @classmethod
    def from_h5(cls, path):
        f = h5py.File(path, "r"); cfg = json.loads(f.attrs["model_config"])
        g = f["model_weights"]; L = []
        for lay in cfg["config"]["layers"]:
            c, n = lay["class_name"], lay["config"]["name"]
            if c == "LSTM":
                L.append({"type": "lstm", "u": c_u(lay), "act": lay["config"]["activation"],
                          "ract": lay["config"]["recurrent_activation"],
                          "W": g[n][n]["kernel:0"][()], "U": g[n][n]["recurrent_kernel:0"][()],
                          "b": g[n][n]["bias:0"][()]})
            elif c == "Dense":
                L.append({"type": "dense", "act": lay["config"]["activation"],
                          "W": g[n][n]["kernel:0"][()], "b": g[n][n]["bias:0"][()]})
            elif c == "Dropout":
                pass                                      # identity at inference
        return cls(L)
    @classmethod
    def from_keras(cls, model):
        L = []
        for lay in model.layers:
            c = lay.__class__.__name__; cfg = lay.get_config()
            if c == "LSTM":
                W, U, b = lay.get_weights()
                ra = cfg["recurrent_activation"]
                ra = "hard_sigmoid" if not isinstance(ra, str) or "hard_sigmoid_k2" in ra else ra
                L.append({"type": "lstm", "u": cfg["units"], "act": cfg["activation"],
                          "ract": ra, "W": W, "U": U, "b": b})
            elif c == "Dense":
                W, b = lay.get_weights()
                L.append({"type": "dense", "act": cfg["activation"], "W": W, "b": b})
        return cls(L)
    def predict(self, x3):
        """x3: (n, 3) -> (n,), batch over independent inputs."""
        x = np.asarray(x3, float)
        for lay in self.layers:
            if lay["type"] == "lstm":
                u = lay["U"].shape[0]; n = x.shape[0]
                h = np.zeros((n, u)); c = np.zeros((n, u))
                fa, ra = _act(lay["act"]), _act(lay["ract"])
                for t in range(x.shape[1]):
                    z = x[:, t:t + 1] @ lay["W"] + h @ lay["U"] + lay["b"]
                    i, f_, g_, o = (ra(z[:, :u]), ra(z[:, u:2 * u]), fa(z[:, 2 * u:3 * u]), ra(z[:, 3 * u:]))
                    c = f_ * c + i * g_; h = o * fa(c)
                x = h
            else:
                x = _act(lay["act"])(x @ lay["W"] + lay["b"])
        return x[:, 0]
    def macs(self):
        m = 0
        for lay in self.layers:
            if lay["type"] == "lstm":
                u = lay["U"].shape[0]; m += 4 * u * (lay["W"].shape[0] + u) * 3   # 3 time steps
            else:
                m += lay["W"].shape[0] * lay["W"].shape[1]
        return m
    def n_params(self):
        return int(sum(np.size(v) for lay in self.layers for k, v in lay.items() if k in ("W", "U", "b")))


def c_u(lay): return lay["config"]["units"]


def run_published(test_data, model, emax):
    """Exactly run_exp() for employ_update_policy = False. Returns dict."""
    x = np.asarray(test_data, float)
    lst = [x[0]] + [x[i + 1] - x[i] for i in range(3)]       # initialization_period = 3
    temp = float(np.sum(lst)); agg = list(lst[1:]); miss = len(lst)
    tx = np.zeros(len(x), bool); tx[:4] = True
    recon = x.copy()
    def step(i, y_pred):
        nonlocal temp, miss
        new_temp = temp + y_pred
        if abs(x[i] - new_temp) < emax:
            agg.append(y_pred); temp = temp + y_pred; recon[i] = temp
        else:
            agg.append(x[i] - temp); temp = x[i]; miss += 1; tx[i] = True
    for i in range(4, 24):                                    # LMS phase, LMS_period = 20
        step(i, float(LMS_W @ np.array(agg[-3:])))
    for i in range(24, len(x)):                               # LSTM phase
        step(i, float(model.predict(np.array(agg[-3:])[None, :])[0]))
    return {"reduction_pct": 100 * (1 - miss / len(x)), "miss": miss, "tx": tx, "recon": recon}
