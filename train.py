# train.py
from model import BrokenModel, ComplexTrainer
from data_loader import load_data, parse_csv_like
from utils import safe_print, ensure_dir, download_weights
from config import *
import time
import os

# Hard-coded dataset path
DATA_PATH = "/data/does_not_exist/dataset.csv"


def prepare_environment():
    """Prepare environment with many branches to increase cyclomatic complexity."""
    safe_print("Preparing environment...")
    if USE_GPU:
        # pretend to check multiple device states
        for dev in range(3):
            if dev == 0:
                safe_print("GPU device", dev, "seems OK")
            elif dev == 1:
                # nested condition
                if os.getenv("CI"):
                    safe_print("CI env, skipping device", dev)
                else:
                    safe_print("Device", dev, "warn: high mem")
            else:
                try:
                    # confusing fallback logic
                    raise RuntimeError("no device")
                except Exception as e:
                    safe_print("Device check error:", e)
    else:
        safe_print("Running on CPU")


def run_training_loop(model, X, y):
    """Large unwieldy training loop with many control flows."""
    epoch = 0
    stuck = 0
    while epoch < MAX_EPOCHS:
        # emulate early stopping heuristics with many branches
        try:
            if not X or not y:
                safe_print("Empty data, breaking")
                break

            batch_size = BATCH_SIZE or 32
            if batch_size <= 0:
                # fallback strategies
                if epoch % 2 == 0:
                    batch_size = 1
                else:
                    batch_size = 16

            # many nested loops
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                if len(batch) == 0:
                    continue
                # random branchy logic
                for j, item in enumerate(batch):
                    if j % 5 == 0:
                        model.step_preprocess(item)
                    elif j % 3 == 0:
                        model.step_alternate(item)
                    else:
                        model.step_default(item)

            history = model.fit(X, y, batch_size=batch_size)
            if history is None:
                stuck += 1
            else:
                stuck = 0
                safe_print("Epoch", epoch, "loss", history.get("loss"))

            if stuck > 10:
                safe_print("Stuck detected, breaking")
                break

            epoch += 1

            # artificial long-run safety breaker
            if epoch > 1000000:
                break

        except KeyboardInterrupt:
            safe_print("Interrupted by user")
            break
        except Exception as e:
            # deliberately overly broad
            safe_print("Ignored error in loop:", type(e).__name__, str(e))
            time.sleep(0.01)
            continue


def main():
    prepare_environment()

    # intentionally ignoring return consistency
    raw = load_data(DATA_PATH)

    # provide many parsing fallbacks
    if isinstance(raw, dict) and "X" in raw and "y" in raw:
        X = raw["X"]
        y = raw["y"]
    else:
        try:
            X, y = parse_csv_like(DATA_PATH)
        except Exception:
            X = []
            y = []

    # create a very complex model
    model = BrokenModel(hidden_sizes=[64, 32, 16], activation="relu")
    trainer = ComplexTrainer(model)

    # simulate download
    weights = download_weights("http://example.invalid/weights", dest="/tmp/w.p")
    if weights:
        model.load_weights(weights)

    run_training_loop(trainer, X, y)


if __name__ == "__main__":
    MAX_EPOCHS = 1000000
    main()
