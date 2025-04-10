from collections import defaultdict
import os.path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_log_error
from sklearn.metrics.pairwise import cosine_similarity

import argparse
from scipy.stats import entropy
from entity.Trace import Trace
from helper import hist_metric
from helper import io_util
from helper.scaler import min_max_scaler


def ab_proportion(decisionDf: pd.DataFrame, labelDf: pd.DataFrame):
    """
    calculate the proportion of anomaly and normal in sampled traces

    Args
    -------
    decisionDf: dataframe [traceId, decision]
    labelDf: dataframe [traceId, label]

    Returns
    -------
    The proportion of abnormal
        and normal traces
    """
    df = pd.merge(decisionDf, labelDf)
    sampleDf = df[df['decision'] == True]
    ab_prop = len(sampleDf[sampleDf['label'] == 1]) / len(sampleDf)
    norm_prop = len(sampleDf[sampleDf['label'] == 0]) / len(sampleDf)
    return round(ab_prop, 3), \
        round(norm_prop, 3)


def typeStatistic(decisionDf: pd.DataFrame, typeDf: pd.DataFrame):
    """
    calculate the total number and coverage
     of type in sampled traces

    Args
    -------
    decisionDf: dataframe [traceId, decision]
    labelDf: dataframe [traceId, label, isError]
    typeDf: dataframe [traceId, pathCode, pathId]

    Returns
    -------
    The nunique of abnormal
        and normal traces
    """
    df = pd.merge(decisionDf, typeDf, on="traceId")
    sampleDf = df[df['decision'] == True]
    sample_n_type = sampleDf['pathCode'].nunique()
    n_type = df['pathCode'].nunique()
    return sample_n_type, round(sample_n_type/n_type, 3)


def diversity(decisionDf: pd.DataFrame, 
              labelDf: pd.DataFrame, 
              typeDf: pd.DataFrame):
    """
    calculate the number of normal or abnormal trace
    types that are included in the sampled traces.

    Args
    -------
    decisionDf: dataframe [traceId, decision]
    labelDf: dataframe [traceId, label]
    typeDf: dataframe [traceId, type]

    Returns
    -------
    the number of normal or abnormal trace
    types in the sampled traces.
    """
    df = pd.merge(decisionDf, labelDf, on="traceId")
    df = pd.merge(df, typeDf, on="traceId")
    sampleDf = df[df['decision'] == True]
    ab_con = sampleDf['label'] == 1
    ab_type_nunique = sampleDf[ab_con]['pathCode'].nunique()
    norm_type_nunique = sampleDf[~ab_con]['pathCode'].nunique()
    return ab_type_nunique, norm_type_nunique



def std(decisionDf: pd.DataFrame, typeDf: pd.DataFrame):
    """
    calculate the coefficient of variation for sampled types
    (updated from sifter)

    Args:
        decisionDf: dataframe [traceId, decision]
        typeDf: dataframe [traceId, pathCode, pathId]

    Returns:
    std float
    cv float
    """
    df = pd.merge(decisionDf, typeDf, on="traceId")
    all_types = df['pathCode'].unique().tolist()

    sampleDf = df[df['decision'] == True]

    sample_dist = []
    for t in all_types:
        sample_num = len(sampleDf[sampleDf['pathCode'] == t])
        sample_dist.append(sample_num)

    std = np.std(sample_dist)
    mu = np.mean(sample_dist)
    cv = std/mu
    return round(std, 3), round(cv, 3)



def recall(decisionDf: pd.DataFrame, 
           labelDf: pd.DataFrame, 
           typeDf: pd.DataFrame):
    """
    evaluate the recall for abnormal types

    Args
    -------
    decisionDf: dataframe [traceId, decision]
    labelDf: dataframe [traceId, label, isError]
    typeDf: dataframe [traceId, pathCode, pathId]

    Returns
    -------
    recall.
    """
    df = pd.merge(decisionDf, labelDf, on="traceId")
    df = pd.merge(df, typeDf, on="traceId")

    ab_df = df[df['label']==1].copy()
    ab_df['isError'] = ab_df['isError'].astype('str')
    ab_df['pathCode'] = ab_df['pathCode'].astype('str')
    ab_df['ab_type'] = ab_df['pathCode'].str.cat(ab_df['isError'], sep='-')

    recall_ab_types = ab_df[ab_df['decision']==True]['ab_type'].nunique()
    all_ab_types = ab_df['ab_type'].nunique()
    recall_pd_types = ab_df[(ab_df['decision']==True) & (ab_df['isError']=='0')]['ab_type'].nunique()
    all_pd_types = ab_df[ab_df['isError']=='0']['ab_type'].nunique()
    recall_error_types = ab_df[(ab_df['decision']==True) & (ab_df['isError']=='1')]['ab_type'].nunique()
    all_error_types = ab_df[ab_df['isError']=='1']['ab_type'].nunique()

    total_recall =  recall_ab_types / all_ab_types
    pd_recall =  recall_pd_types / all_pd_types
    err_recall =  recall_error_types / all_error_types
    return total_recall, pd_recall, err_recall



def mse_percentage_eval(decisionDf: pd.DataFrame, 
            originDf: pd.DataFrame):
    """
    evaluate the mse of several quantiles between sampled and original duration

    Args:
        decisionDf (pd.DataFrame):  [traceId, decision]
        originDf (pd.DataFrame): original data

    Returns:
        mse
    """
    originDf['label'] = originDf['service'] + ':' + originDf['operation']
    sampleIds =  decisionDf.loc[decisionDf['decision'] == True, 'traceId']
    sampleDf = originDf[originDf['traceId'].isin(sampleIds)]
    
    mss = set(originDf['label'])

    mses = []
    ps = [0, 25, 50, 75, 90, 95, 99, 100]
    for ms in mss:
        origin_data = originDf.loc[originDf['label']==ms, 'duration'].values
        sample_data = sampleDf.loc[sampleDf['label']==ms, 'duration'].values

        if len(sample_data) == 0:
            sampleP = np.array([min(origin_data)] * len(ps))
        else:
            sampleP = np.nanpercentile(sample_data, ps)
        originP = np.nanpercentile(origin_data, ps)

        maxV, minV = max(origin_data), min(origin_data)
        sampleP = (sampleP - minV) / (maxV - minV + (1e-7))
        originP = (originP - minV) / (maxV - minV + (1e-7))

        mse = np.mean((sampleP- originP)**2)
        mses.append(mse)

    return round(np.sum(mses), 3)



def RoD(decisionDf: pd.DataFrame, sampleRate: float):
    """
    evaluate the difference between the target number
    and the sampled number

    Args
    -------
    decisionDf: dataframe [traceId, decision]
    sampleRate: float

    Returns
    -------
    difference ratio
    """
    targetNum = int(sampleRate * len(decisionDf))
    sampledNum = len(decisionDf[decisionDf['decision']==True])
    diffRatio = round(abs(sampledNum - targetNum) / targetNum, 3)
    return diffRatio


def pkl2df(traces: List[Trace]):
    spanIds, traceIds, durations, services, operations = \
        [],[],[],[],[]
    for trace in traces:
        for span in trace.spans:
            spanIds.append(span.spanId)
            durations.append(span.duration)
            services.append(span.service)
            operations.append(span.operation)
            traceIds.append(span.traceId)
    return pd.DataFrame(data={
        "spanId": spanIds,
        "traceId": traceIds,
        "service": services,
        "operation": operations,
        "duration": durations
    })


parser = argparse.ArgumentParser()
parser.add_argument('--saveDir', type=str, default='output')
parser.add_argument('--dataDir', type=str, default='data')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--dataSet', type=str, default='trainticket')
parser.add_argument('--sampleRate', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

if __name__ == "__main__":
    methods = args.methods
    dataSet = args.dataSet
    dataDir = args.dataDir
    saveDir = args.saveDir
    sampleRate = args.sampleRate

    os.makedirs(f"{saveDir}/res", exist_ok=True)
    eval_log = open(f'{saveDir}/res/{dataSet}-result.csv', 'w+')
    eval_log.write("method,count,RoD,typeNum,coverage,abTypeNum,\
                   normTypeNum,std,CV,Recall,Recall_pd, Recall_err,MSE,\
                   encodeT, sampleT, otherT, totalT\n")

    originPkl: List[Trace] = io_util.load(f'data/{dataSet}/traces.pkl')
    originDf = pkl2df(originPkl)
    for method in methods:
        ab_prop, norm_prop, \
        type_num, type_cov, \
        ab_type_num, norm_type_num, \
        rec,rec_pd, rec_err, true_ratio_norm = None,None,None,None,None,None,None,None,None,None

        decisionDf = pd.read_csv(f'{saveDir}/{dataSet}-{method}-sample.csv')
        costDf = pd.read_csv(f'{saveDir}/{dataSet}-{method}-cost.csv')
        typeDf = pd.read_csv(f'{dataDir}/{dataSet}/type.csv')

        label_pth = f'{dataDir}/{dataSet}/new-labels.csv'
        if os.path.exists(label_pth):
            labelDf = pd.read_csv(label_pth)
            ab_type_num, norm_type_num = diversity(decisionDf, labelDf, typeDf)
            rec, rec_pd, rec_err = recall(decisionDf, labelDf, typeDf)

        sample_n = len(decisionDf[decisionDf['decision']==True])
        type_num, type_cov = typeStatistic(decisionDf, typeDf)
        stdVal, cvVal = std(decisionDf, typeDf)
        # KS_mean, true_ratio = ks_eval(decisionDf, originDf, None, False)
        mse = mse_percentage_eval(decisionDf, originDf)

        rod = RoD(decisionDf, sampleRate)
        encode_cost = costDf['encode_t'].mean()
        sample_cost = costDf['sample_t'].mean()
        other_cost = costDf['other_t'].mean()
        total_cost = costDf['total_t'].mean()

        eval_log.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            method, sample_n, rod, type_num, type_cov, ab_type_num,
            norm_type_num, stdVal, cvVal, rec,rec_pd, rec_err, mse,
            encode_cost, sample_cost, other_cost,total_cost
        ))
