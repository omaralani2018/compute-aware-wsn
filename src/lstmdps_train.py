"""
LSTM-DPS training as in the released code (utils/_model.py), in TensorFlow 2:
  1. hyperopt TPE over the released search space, hyperopt_max_trials = 5,
     loss 'mse', model chosen by minimum validation loss (70/30 split, no shuffle);
  2. number_random_fits = 2 fits of the chosen architecture with loss 'mae',
     checkpoint on best validation MAE, keep the fit with the lower validation loss;
  3. 'refit' (Custom_Model.update): 2 further epochs on the initialization +
     LMS-phase aggregate list, 70/30 split, the in-memory model is deployed.
Settings are those recorded in the authors' saved Intel experiments.
"""
import os, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

SPACE = {
    'nb_epochs': hp.choice('nb_epochs', [10]),
    'lstm_cells': hp.choice('lstm_cells', [8, 16]),
    'd1_nodes': hp.choice('d1_nodes', [4, 8]),
    'd2_nodes': hp.choice('d2_nodes', [0, 4]),
    'batch_size': hp.choice('batch_size', [4, 8, 16]),
    'dropout1': hp.choice('dropout1', [0, 0.1, 0.2]),
    'activation_1': hp.choice('activation_1', ['relu', 'linear']),
    'activation_2': hp.choice('activation_2', ['relu', 'linear']),
}
def hard_sigmoid_k2(z):
    return tf.clip_by_value(0.2 * z + 0.5, 0.0, 1.0)


W = 3; RATIO = 0.7; TRIALS = 5; FITS = 2


def look_back(series, diff=True):
    """get_data_look_back(): differences, W lag features, next difference as target."""
    s = np.asarray(series, float)
    d = np.diff(s) if diff else s
    X = np.stack([d[i:len(d) - W + i] for i in range(W)], 1)
    y = d[W:]
    return X[..., None], y


def split(X, y):
    n = int(X.shape[0] * RATIO)
    return X[:n], y[:n], X[n:], y[n:]


def make(p, keras_seed):
    keras.utils.set_random_seed(int(keras_seed))
    inp = layers.Input((W, 1))
    # Keras 2.2.4 (the released models): recurrent_activation = hard_sigmoid
    # = clip(0.2 z + 0.5, 0, 1). Keras 3 redefined 'hard_sigmoid' with slope 1/6,
    # so the Keras 2 function is passed explicitly.
    o = layers.LSTM(int(p['lstm_cells']), recurrent_activation=hard_sigmoid_k2)(inp)
    o = layers.Dense(int(p['d1_nodes']), activation=p['activation_1'], name='denes_1')(o)
    if int(p['d2_nodes']):
        o = layers.Dense(int(p['d2_nodes']), activation=p['activation_2'], name='denes_2')(o)
    if p['dropout1']:
        o = layers.Dropout(p['dropout1'])(o)
    o = layers.Dense(1)(o)
    return keras.Model(inp, o)


def build(historical, seed):
    X, y = look_back(historical, diff=True)
    Xt, yt, Xv, yv = split(X, y)
    rs = np.random.default_rng(seed)
    def objective(p):
        m = make(p, rs.integers(1 << 30)); m.compile(loss='mse', optimizer='adam', metrics=['mae'])
        h = m.fit(Xt, yt, batch_size=int(p['batch_size']), epochs=int(p['nb_epochs']),
                  verbose=0, validation_data=(Xv, yv), shuffle=False)
        return {'loss': float(np.min(h.history['val_loss'])), 'status': STATUS_OK}
    best = fmin(objective, SPACE, algo=tpe.suggest, max_evals=TRIALS, trials=Trials(),
                rstate=np.random.default_rng(seed), show_progressbar=False)
    p = space_eval(SPACE, best)
    fits = []
    for _ in range(FITS):
        m = make(p, rs.integers(1 << 30)); m.compile(loss='mae', optimizer='adam', metrics=['mae'])
        best_w, best_v = None, np.inf
        for ep in range(int(p['nb_epochs'])):            # ModelCheckpoint(save_best_only, val MAE)
            h = m.fit(Xt, yt, batch_size=int(p['batch_size']), epochs=1, verbose=0,
                      validation_data=(Xv, yv), shuffle=False)
            v = h.history['val_mae'][0]
            if v < best_v: best_v, best_w = v, m.get_weights()
        m.set_weights(best_w); fits.append((best_v, m))
    fits.sort(key=lambda t: t[0])
    return fits[0][1], p


def refit(model, p, agg_list):
    """Custom_Model.update(): Apply_diff=False on the aggregate list of differences."""
    X, y = look_back(agg_list, diff=False)
    Xt, yt, Xv, yv = split(X, y)
    # the model is still compiled with loss 'mae' and Adam from build(), as in the
    # released code, so training simply continues for two epochs
    model.fit(Xt, yt, batch_size=int(p['batch_size']), epochs=2, verbose=0,
              validation_data=(Xv, yv), shuffle=False)
    return model
