# -*- coding: utf-8 -*-
"""

"""
import random
import numpy as np
import geatpy as ea
import time


np.set_printoptions(suppress=True)

class SampleProblem2(ea.Problem):

    def __init__(self, 
                 rawDist: list,
                 abDist: list,
                 quotas: list, 
                 bases: list,
                 combCount: int,
                 M=1):
        """
        Args:
            dim (int): decision variable dimensions
            rawDist (list):  latency distributio of candidate data
            abDist (list):  latency distribution of abnormal data
            quotas (list): sampling quota of each code
            bases (list): storaged count of each code
            idx2label (dict): (idx, operation)
            randomHist (defaultdict): latency histogram of random sampling
            combCount (int): count of combinations for each code
            M (int, optional): count of objectives. Defaults to 1.
        """
        if combCount < 2:
            raise Exception("combCount must be larger than 2")

        self.quotas = np.array(quotas)
        self.bases = np.array(bases)
        self.rawDist = np.array(rawDist) # (numTrace, numLabel)
        self.abDist = np.array(abDist) # (numAbTrace, numLabel)

        self.splits = np.cumsum(self.bases)
        self.C = np.sum(self.quotas)
        
        self.numLabel = self.rawDist.shape[1]

        # initiate the combinations for each code
        init_st = time.time()
        i = 0
        self.allCombs = []
        for start, end in zip([0] + list(self.splits[:-1]), self.splits):
            combs = [Combination(random.sample(range(start, end), self.quotas[i]))
                     for _ in range(combCount)]
            self.allCombs.append(combs)
            i+=1
        self.allCombs = np.array(self.allCombs).T # (combCount, numCode)
        init_et = time.time()
        print(f"[SAMPLE] the time cosuming of init is {(init_et-init_st):.2f} seconds")

        name = 'SampleProblem'
        maxormins = [1] # -1: maximize; 1: minimize
        Dim = len(quotas)
        self.Dim = Dim
        varTypes = [1] * Dim # 0: continuous; 1: discrete
        lb = [0] * Dim
        ub = [combCount-1] * Dim
        lbin = [1] * Dim  # 1: include lower bound
        ubin = [1] * Dim  # 1: include upper bound
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)
        
        # calculate percentage data of raw data
        self.ps = [0, 25, 50, 75, 90, 95, 99, 100]
        if len(self.abDist) > 0:
            origin = np.vstack((self.rawDist, self.abDist)).T # (numLabel, numTrace)
        else:
            origin = self.rawDist.T # (numLabel, numTrace)
        self.originP = np.nanpercentile(
            origin, 
            self.ps, 
            axis=1).T # (numLabel, 8)
        
        self.maxV = np.nanmax(origin, axis=1).reshape(-1,1)
        self.minV = np.nanmin(origin, axis=1).reshape(-1,1)

        # scaler originP
        self.originP = (self.originP - self.minV) / (self.maxV-self.minV+(1e-7))

    def randomPhen(self, n):
        Phen = np.random.randint(0,self.allCombs.shape[0], size=(n, self.allCombs.shape[1]))
        return np.array(Phen)
    

    def consistency(self, matrix):
        # matrix: (Np * numLabel, C)
        abDistT = self.abDist.T # (numLabel, numAbTraces)
        if len(self.abDist) > 0:
            tileAbDist = np.tile(abDistT, (self.Np, 1)) # (Np * numLabel, numAbTraces)
            sample = np.hstack((matrix, tileAbDist))
        else:
            sample = matrix

        sampleP = np.nanpercentile(sample, self.ps, axis=1).T
        originP = np.tile(self.originP, (self.Np, 1)) # (Np * numLabel, 8)

        # scaler
        minV = np.tile(self.minV, (self.Np, 1))
        maxV = np.tile(self.maxV, (self.Np, 1))
        sampleP = (sampleP - minV) / (maxV-minV+(1e-7))

        # calculate RMSE
        mse = np.mean((sampleP - originP)**2, axis=1)
        
        # res: (Np * numLabel, 1)
        res = np.array(mse).reshape(self.Np, -1) # res: (Np, numLabel)
        return np.sum(res, axis=1)


    def evalVars(self, Vars):
        selectCombs = self.allCombs[Vars, range(self.allCombs.shape[1])]
        extractFunc = np.frompyfunc(lambda a: a.comb, 1, 1)
        selectIdxs = np.array([np.concatenate(row, axis=None) for row in extractFunc(selectCombs)]).astype(int) # (Np, C)
        sampleData = self.rawDist[selectIdxs, :] # (Np, C, numLabel)
        sampleData = np.transpose(sampleData, (0, 2, 1)) # (Np, numLabel, C)
        sampleData = sampleData.reshape(-1, sampleData.shape[-1]) # (Np * numLabel, C)
        
        self.Np =Vars.shape[0] 
        
        # evolution
        fs = self.consistency(sampleData).reshape(-1,1)
                
        return fs
    

    def getIdxsByVar(self, var):
        selectCombs = self.allCombs[var, range(self.allCombs.shape[1])]
        extractFunc = np.frompyfunc(lambda a: a.comb, 1, 1)
        selectIdxs = np.array([np.concatenate(row, axis=None) for row in extractFunc(selectCombs)]).astype(int)
        return selectIdxs.flatten().tolist()


class Combination:
    def __init__(self, comb) -> None:
        self.comb = np.array(comb)