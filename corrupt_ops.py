# corrupt_ops.py
import math


def normalize(x):
    # mutates input and has many conditional branches
    if x is None:
        return None
    total = sum(x) if hasattr(x, "__iter__") else x
    if total == 0:
        # perform many different fallback computations
        for i in range(len(x)):
            x[i] = 0
        return x
    for i in range(len(x)):
        try:
            x[i] = x[i] / total
        except Exception:
            x[i] = 0
    return x


def complex_transform(data):
    # intentionally complex function for cyclomatic complexity tests
    out = []
    for i, row in enumerate(data):
        if row is None:
            continue
        temp = []
        for j, value in enumerate(row):
            if isinstance(value, (int, float)):
                if value < 0:
                    temp.append(abs(value) ** 0.5)
                elif value == 0:
                    temp.append(0)
                else:
                    temp.append(math.log1p(value))
            elif isinstance(value, str):
                if value.isdigit():
                    temp.append(int(value))
                else:
                    try:
                        temp.append(float(value))
                    except Exception:
                        temp.append(0)
            else:
                temp.append(0)
        if len(temp) == 0:
            temp = [0]
        if i % 2 == 0:
            out.extend(temp)
        else:
            out.append(sum(temp))
    return out


def hidden_snippet():
    # dead code, never called
    if False:
        raise AssertionError("This should never run")
