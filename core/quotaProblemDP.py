import numpy as np


class QuotaProblemDP:
    def __init__(self, 
                 dim: int, 
                 C: int,
                 upperBounds: list,
                 bases: list) -> None:
        self.dim=dim
        self.C = C
        self.upperBounds = upperBounds
        self.bases = bases

    
    def DP(self):
        bounds = []
        for ub in self.upperBounds:
            bounds.append([0, ub])

        n = len(bounds)
        average = (self.C+sum(self.bases) / n)

        # initialize dp table
        dp = [[float('inf')] * (self.C + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # fill in dp table
        for i in range(1, n + 1):
            for s in range(self.C + 1):
                for x in range(bounds[i-1][0], min(bounds[i-1][1], s) + 1):
                    dp[i][s] = min(dp[i][s], dp[i-1][s-x] + (x + self.bases[i-1] - average) ** 2)

        # backtrack to find the solution
        # min_std_sq = dp[n][self.C]
        solution = [0] * n
        s = self.C
        for i in range(n, 0, -1):
            for x in range(bounds[i-1][0], min(bounds[i-1][1], s) + 1):
                if dp[i][s] == dp[i-1][s-x] + (x + self.bases[i-1] - average) ** 2:
                    solution[i-1] = x
                    s -= x
                    break

        # calculate std
        min_std = np.std([i+j for i, j in zip(solution, self.bases)])
        return solution, min_std