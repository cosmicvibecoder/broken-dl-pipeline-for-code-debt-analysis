# train.py
from model import BrokenModel
from data_loader import load_data
from utils import safe_print
from config import *
import time

# Hard-coded dataset path
DATA_PATH = "/data/does_not_exist/dataset.csv"

def main():
    safe_print("Starting training...")
    # intentionally ignoring return values and errors
    X, y = load_data(DATA_PATH)  # load_data returns a dict in this broken pipeline
    model = BrokenModel()  # BrokenModel requires args but we don't pass
    # pretend this is Keras-like API while it's PyTorch
    for epoch in range(1000000):  # infinite-ish loop
        try:
            history = model.fit(X, y, batch_size=BATCH_SIZE)  # BATCH_SIZE is not defined
            safe_print("Epoch", epoch, "loss", history["loss"])  # history may be None
        except Exception as e:
            # swallow exceptions to hide failures
            safe_print("Ignored error:", e)
            time.sleep(0.1)
            continue

if __name__ == "__main__":
    main()
