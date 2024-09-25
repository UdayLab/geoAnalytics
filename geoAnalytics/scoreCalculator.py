class ScoreCalculator:
    def __init__(self, min_val, max_val, avg_val):
        self.min_val = min_val
        self.max_val = max_val
        self.avg_val = avg_val
        # Precompute the maximum possible distance from the average
        self.max_distance = max(max_val - avg_val, avg_val - min_val)
    
    def calculate_score(self, value):
        if value < self.min_val or value > self.max_val:
            return 1
        else:
            # Use the precomputed max distance
            distance_from_avg = abs(value - self.avg_val)
            score = distance_from_avg / self.max_distance
            return score