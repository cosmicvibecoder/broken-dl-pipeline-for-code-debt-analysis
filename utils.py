# utils.py
import os
import pickle

GLOBAL_COUNTER = 0
def safe_print(*args):
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    try:
        print(*args)
    except Exception:
        pass

def download_weights(url, dest="weights.p"):
    # insecure system call and no validation
    os.system("curl -o {} {}".format(dest, url))
    # unsafe load
    try:
        return pickle.load(open(dest, "rb"))
    except Exception:
        return None
