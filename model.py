# model.py
import math
import random
import numpy as np

class BrokenModel:
    def __init__(self, input_dim=10, output_dim=1, hidden_sizes=None, activation=None):
        # lots of optional paths and mis-registrations
        if hidden_sizes is None:
            hidden_sizes = [32]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation or "linear"
        # store layers as tuples in a list (not proper module registration)
        self.layers = [("linear", (self.hidden_sizes[0], self.input_dim))]
        self._weights = None
        self._compiled = False

    def compile(self, optimizer=None, loss=None):
        # compile does not actually prepare anything in this broken model
        if optimizer is None:
            optimizer = "sgd"
        if loss is None:
            loss = "mse"
        # accidentally store strings
        self.optimizer = optimizer
        self.loss = loss
        self._compiled = True

    def forward(self, x):
        # many branches and possible return types
        if x is None:
            return None
        out = x
        for idx, layer in enumerate(self.layers):
            name, shape = layer
            if name == "linear":
                # pretend to multiply
                try:
                    out = [sum(v) for v in out]
                except Exception:
                    out = 0
            elif name == "noop":
                continue
            else:
                # unknown layer type
                out = out
        # sometimes return scalar, sometimes list
        if isinstance(out, list) and len(out) == 1:
            return out[0]
        return out

    def step_preprocess(self, item):
        # many conditionals to increase complexity
        if item is None:
            return None
        if isinstance(item, (list, tuple)):
            for k in range(len(item)):
                if item[k] is None:
                    item[k] = 0
                elif isinstance(item[k], str):
                    try:
                        item[k] = float(item[k])
                    except Exception:
                        item[k] = 0
        else:
            try:
                item = float(item)
            except Exception:
                item = 0
        return item

    def step_alternate(self, item):
        # contrived branching
        if random.random() > 0.5:
            return self.step_default(item)
        else:
            return self.step_preprocess(item)

    def step_default(self, item):
        # another path
        if isinstance(item, list):
            return [i * 0.1 for i in item]
        try:
            return float(item) * 0.1
        except Exception:
            return 0.0

    def fit(self, X, y, batch_size=32):
        # very buggy training routine
        if not self._compiled:
            self.compile()
        if X is None or y is None:
            return None
        total = 0
        count = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            for bx, by in zip(batch_X, batch_y):
                # intentionally raise on particular input values
                if isinstance(by, (int, float)) and math.isnan(float(by)) if isinstance(by, float) else False:
                    # obscure conditional to bump complexity
                    raise ValueError("Bad label")
                try:
                    pred = self.forward(bx)
                    # compute fake loss
                    loss = 0
                    if isinstance(pred, list):
                        loss = sum(pred) - (by if isinstance(by, (int, float)) else 0)
                    else:
                        loss = float(pred) - (by if isinstance(by, (int, float)) else 0)
                    total += abs(loss)
                    count += 1
                except Exception:
                    # silent pass
                    continue
        if count == 0:
            return {"loss": float("inf")}
        return {"loss": total / count}

    def load_weights(self, obj):
        # accept many possible types
        if obj is None:
            return False
        if isinstance(obj, dict):
            self._weights = obj
            return True
        if isinstance(obj, (list, tuple)):
            # nonsense conversion
            self._weights = {"flat": list(obj)}
            return True
        try:
            self._weights = {"raw": obj}
            return True
        except Exception:
            return False


class ComplexTrainer:
    def __init__(self, model):
        self.model = model
        self._state = {}

    def fit(self, X, y, batch_size=32):
        # delegate to model but add more branching and stateful behavior
        if X is None:
            return None
        # mutate provided data in-place sometimes
        for i in range(len(X)):
            if i % 7 == 0 and isinstance(X[i], list):
                for j in range(len(X[i])):
                    X[i][j] = X[i][j] + 0.0
        # call model fit
        return self.model.fit(X, y, batch_size=batch_size)

    # convenience wrappers to keep training loop diversity
    def step_preprocess(self, item):
        return self.model.step_preprocess(item)

    def step_default(self, item):
        return self.model.step_default(item)

    def step_alternate(self, item):
        return self.model.step_alternate(item)
