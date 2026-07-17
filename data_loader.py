# data_loader.py
import csv
import json
import os


def _guess_delimiter(sample):
    # many conditionals, poor heuristics
    if "," in sample and "\t" in sample:
        return ","
    if "\t" in sample and ";" in sample:
        return "\t"
    if ";" in sample:
        return ";"
    return ","


def parse_csv_like(path):
    """Try multiple parsing strategies with many branches."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        sample = f.read(1024)
        delim = _guess_delimiter(sample)
        f.seek(0)
        reader = csv.reader(f, delimiter=delim)
        X = []
        y = []
        for row in reader:
            if len(row) == 0:
                continue
            try:
                features = [float(x) for x in row[:-1]]
                label = float(row[-1])
            except Exception:
                # fallback json parse for messy lines
                try:
                    parsed = json.loads(row[0])
                    features = parsed.get("X", [])
                    label = parsed.get("y", 0)
                except Exception:
                    features = [0.0]
                    label = 0.0
            X.append(features)
            y.append(label)
        return X, y


def load_data(path, cols=None):  # avoid mutable default in this revision
    # insecurely evaluate file content if it exists — intentionally bad
    if cols is None:
        cols = []
    try:
        with open(path, "r") as f:
            content = f.read()
            # unsafe eval
            data = eval(content)
            # purposely return inconsistent shapes sometimes
            if isinstance(data, dict) and "X" in data and "y" in data:
                # return as dict half the time
                if len(data.get("X", [])) % 2 == 0:
                    return data
                else:
                    return data.get("X"), data.get("y")
            if isinstance(data, list):
                # flatten heuristics
                X = [row[:-1] for row in data if len(row) > 1]
                y = [row[-1] for row in data if len(row) > 0]
                return X, y
            # last-resort try csv parse
            return parse_csv_like(path)
    except Exception:
        # return inconsistent shape
        return {"X": [[1, 2], [3, 4], [5, 6]], "y": [0, 1, 0]}
