import argparse
import os
import random
import numpy as np
import pandas as pd
from helper import io_util
from core.TracePicker import TracePicker


parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='data')
parser.add_argument('--dataSet', type=str, default='A')
parser.add_argument('--saveDir', type=str, default='output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--bufferSize', type=int, default=4000)
parser.add_argument('--poolHeight', type=int, default=1000)
parser.add_argument('--combCount', type=int, default=100)
parser.add_argument('--sampleRate', type=float, default=0.1)
args = parser.parse_args()


def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)
    setSeed(args.seed)

    traces = io_util.load(f'{args.dataDir}/{args.dataSet}/traces.pkl')

    picker = TracePicker(bufferSize=args.bufferSize, 
                        poolHeight=args.poolHeight,
                        sampleRate=args.sampleRate,
                        combCount=args.combCount)

    encode_ts, other_ts, sample_ts = [], [], []
    ids = []
    sampleIds = []
    for i, trace in enumerate(traces):
        endSignal = (i==(len(traces)-1))
        ids.append(trace.traceID)
        tmpSpIds, encode_t, other_t, sample_t = picker.accept(trace, endSignal)
        sampleIds.extend(tmpSpIds)
        encode_ts.append(encode_t)
        other_ts.append(other_t)
        sample_ts.append(sample_t)
    
    # for key, value in picker.pathCounter.items():
    #     print(key, value)
    # df = pd.DataFrame(data={
    #     'traceId': picker.traces,
    #     'type': picker.types,
    # })
    # df.to_csv('test.csv')
        
    
    decisions = [id in sampleIds for id in ids]

    encode_cost = sum(encode_ts)
    sample_cost = sum(sample_ts)
    other_cost = sum(other_ts)
    res = pd.DataFrame(data={
        'traceId': ids, 
        'decision': decisions
    })
    cost_res = pd.DataFrame(data={
        'encode_t': [encode_cost],
        'sample_t': [sample_cost],
        'other_t': [other_cost],
        'total_t': [encode_cost+sample_cost+other_cost]
    })
    res.to_csv(f'{args.saveDir}/{args.dataSet}-TracePicker-sample.csv', index=False)
    cost_res.to_csv(f'{args.saveDir}/{args.dataSet}-TracePicker-cost.csv', index=False)