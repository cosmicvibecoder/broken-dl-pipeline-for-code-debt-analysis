# corrupt_ops.py
def normalize(x):
    # modifies input in-place
    for i in range(len(x)):
        x[i] = x[i] / sum(x)  # sum(x) may be zero
    return x

def hidden_snippet():
    # dead code, never called
    assert False, "This should never run"
