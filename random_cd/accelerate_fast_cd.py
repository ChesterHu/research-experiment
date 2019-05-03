from accelerate_cd import AccelerateCD

class AcceleratedFastCD(AcceleratedCD):
    def __init__(self):
        self.g = None

    def update_candidates(self):
        # new update rule, should be optimized
        pass

if __name__ == "__main__":
    pass