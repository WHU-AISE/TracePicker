from collections import Counter, defaultdict
import itertools
import warnings
import numpy as np
import multiprocessing as mp
import random
import time
import geatpy as ea
# from core.quotaProblem import QuotaProblem
from core.quotaProblemDP import QuotaProblemDP
# from core.sampleProblem import SampleProblem
from core.sampleProblem2 import SampleProblem2
from entity.Trace import Trace
from core.buffer import SharedBuffer
from core.bfsEncoder import BFSEncoder
warnings.filterwarnings("ignore")


class TracePicker:
    def __init__(self, 
                 bufferSize, 
                 poolHeight, 
                 sampleRate,
                 combCount) -> None:
        self.limit = bufferSize
        self.buffer = SharedBuffer()
        self.bfsEncoder = BFSEncoder(poolHeight)

        self.pathCounter = Counter()
        self.sampleRate = sampleRate

        # allocate quotas
        self.Np_quota = 1000
        self.Ng_quota = 50

        # Group subset selection problem
        self.Np_sample = 25
        self.Ng_sample = 10
        self.combCount = combCount

        self.abTypes = []
        

    def encode(self, trace: Trace):
        tree, isAb = self.bfsEncoder.buildTreeAndCheck(trace)
        code = self.bfsEncoder.bfsEncode_tree(tree)
        return code, isAb


    def accept(self, 
               trace: Trace, 
               signal: bool):
        encode_t, sample_t, other_t = 0, 0, 0
        ids = []
        # encode the trace
        st = time.time()
        code, isAb = self.encode(trace)
        ed = time.time()
        encode_t = ed - st

        st = ed
        # cache the trace
        self.buffer.add(code, trace, isAb)
        ed = time.time()
        other_t = ed - st

        st = ed
        if (self.limit == self.buffer.count) or signal:
            
            # sample
            ids.extend(self.sample())
            self.buffer.clear()
            self.bfsEncoder.clear()

        ed = time.time()
        sample_t = ed - st
        return ids, encode_t, other_t, sample_t



    def allocateQuota(self, candidatesDict: dict) -> dict:
        """
        allocate sampling quota for paths

        Args:
        -------
            candidatesDict (dict): {code: [id1, id2]}

        Returns:
        -------
            Counter: the quotas for each path
        """
        # codes = list(candidatesDict.keys())
        codes = list(set(list(self.pathCounter.keys())+list(candidatesDict.keys())))
        upperBounds, bases = [], []
        for code in codes:
            if code in candidatesDict.keys():
                upperBounds.append(len(candidatesDict[code]))
            else:
                upperBounds.append(0)
            bases.append(self.pathCounter[code])


        problem = QuotaProblemDP(dim=len(codes), C=self.Ns, upperBounds=upperBounds, bases=bases)
        vars, min_std = problem.DP()
        print(f'[ALLOCATE] The minimum std is: {min_std:.2f}')

        quotas = Counter()
        for code, var in zip(codes, vars):
            if code in candidatesDict.keys():
                quotas[code] = var
        
        return quotas


    def sample(self) -> list:
        """
        sample traces from buffer

        Returns:
        -------
            list: the sampled trace IDs
        """
        ids = []  
        self.Ns = int(self.sampleRate * self.buffer.count)
        codes = self.buffer.getCodes()
        
        groupDist, abDist, candidatesDict = dict(), [], dict()
        allLabels = self.bfsEncoder.getBufferLabels()
        label2idx = {label: idx for idx, label in enumerate(allLabels)}
        # idx2label = {idx: label for idx, label in enumerate(allLabels)}
       
        for code in codes:
            traces = self.buffer.getTracesByCode(code)

            codeDist, candidates = [], []
            for trace in traces:
                # random save hist of durations
                # save_hist = random.random() < self.sampleRate

                durations = [np.nan]*len(allLabels)
                for span in trace.spans:
                    label = span.getSpanLabel()
                    durations[label2idx[label]] = span.duration

                if trace.abnormal and self.Ns > 0:
                    abType = str(code) + '-' + str(trace.isError)
                    if abType in self.abTypes:
                        codeDist.append(durations)
                        candidates.append(trace.traceID)
                    else:
                        self.abTypes.append(abType)
                        ids.append(trace.traceID)
                        self.Ns -= 1
                        abDist.append(durations)
                        self.pathCounter[code]+=1
                else:
                    codeDist.append(durations)
                    candidates.append(trace.traceID)
                # codeDist.append(durations)
                # candidates.append(trace.traceID)

            if len(candidates) > 0:
                candidatesDict[code] = candidates
                groupDist[code] = codeDist

        print(f'[SAMPLE] The count of abnomal traces is {len(ids)}')
        if self.Ns > 0:
            allocate_st = time.time()
            # calculate the quotas for each code
            quotaDict = self.allocateQuota(candidatesDict)

            allocate_ed = time.time()
            print(f'[ALLOCATE] The time consuming is {(allocate_ed-allocate_st):.2f} seconds')

            # Sample Optimization
            optim_st = time.time()
            code2idx = {code: idx for idx, code in enumerate(codes)}
            quotas, bases, dists, candidates = [0]*len(codes), [0]*len(codes), [[]]*len(codes), [[]]*len(codes)
            for code, quota in quotaDict.items():
                quotas[code2idx[code]] = quota
                bases[code2idx[code]] = len(candidatesDict[code])
                dists[code2idx[code]]=groupDist[code]
                candidates[code2idx[code]]=candidatesDict[code]
                self.pathCounter[code]+=quota
            candidates=list(itertools.chain(*candidates))
            dists=list(itertools.chain(*dists))
            
            sampleIds = self.sampleOptim(quotas, dists, abDist, bases, candidates)
            ids.extend(sampleIds)

            optim_ed = time.time()
            print(f'[SAMPLE] The time consuming is {(optim_ed-optim_st):.2f} seconds')
        print(f'[SAMPLE] The sample count is {len(ids)}')
        print(f'=======================================================================')
        return ids


    def sampleOptim(self,
                    quotas: int,
                    dists: list,
                    abDist: list,
                    bases: list,
                    candidates: list):
        problem = SampleProblem2(
            rawDist=dists,
            abDist=abDist,
            quotas=quotas,
            bases=bases,
            combCount=self.combCount,
            M=1,
        )
        # prophetVars = problem.randomPhen(n=1)
        
        # print(f'[SAMPLE] The mse is {problem.evalVars(prophetVars)}')
        # sampleIds = np.array(candidates)[problem.getIdxsByVar(prophetVars)]

        algorithm = ea.soea_DE_best_1_bin_templet(
            problem,
            ea.Population(Encoding="RI", NIND=self.Np_sample),
            MAXGEN=self.Ng_sample,
            logTras=0,
        )

        res = ea.optimize(
            algorithm,
            # prophet=prophetVars,
            verbose=False,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False,
        )

        print(f'[SAMPLE] The MSE is {res["ObjV"].flatten().tolist()[0]}')
        
        idxs = problem.getIdxsByVar(res['Vars'])
        sampleIds = np.array(candidates)[idxs]

        return sampleIds.tolist()