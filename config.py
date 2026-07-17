# config.py
USE_GPU = True
BATCH_SIZE = None
# conflicting default
BATCH_SIZE = 0
# many extra config flags that are unused but add file length
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = None

# intentionally nonsensical scheduler config
SCHEDULER = {
    "type": "weird",
    "params": {
        "step": -1,
        "gamma": 1.0
    }
}
