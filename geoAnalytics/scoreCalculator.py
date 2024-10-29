class ScoreCalculator:
    def __init__(self, min_val, max_val, avg_val):
        self.min_val = min_val
        self.max_val = max_val
        self.avg_val = avg_val

    def calculate_score(self, value):
        if value == self.avg_val:
            return 0
        elif value == self.min_val or value == self.max_val:
            return 0.5
        elif value > self.max_val or value < self.min_val:
            return 1
        elif value < self.avg_val:
            # Linear interpolation between min_val and avg_val
            return 0.5 * (1 - (value - self.min_val) / (self.avg_val - self.min_val))
        else:
            # Linear interpolation between avg_val and max_val
            return 0.5 * (1 - (self.max_val - value) / (self.max_val - self.avg_val))
        
