# data_loader.py
import csv

def load_data(path, cols=[]):  # mutable default arg
    # insecurely evaluate file content if it exists
    try:
        with open(path, "r") as f:
            content = f.read()
            # unsafe eval
            data = eval(content)
            return data  # sometimes a dict not tuple
    except Exception:
        # return inconsistent shape
        return {"X": [[1,2],[3,4]], "y":[0,1]}
