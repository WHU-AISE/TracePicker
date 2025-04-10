import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def bfs_encode(traces: pd.DataFrame, traceId: str):
    trace = traces[traces['traceId'] == traceId]
    root = trace[trace['parentSpanId'] == '-1']['service'].values.tolist()[0]
    edges = trace[['parentService', 'service']].values.tolist()
    dg = nx.DiGraph()
    for edge in edges:
        dg.add_edge(edge[0], edge[1])

    code = str(list(nx.bfs_tree(dg, root)))

    return {'traceId': traceId, 'path': code}


if __name__ == '__main__':
    dataSet = 'A'
    traces = pd.read_csv(f'../data/{dataSet}/sample.csv')
    all_ids = traces['traceId'].values.tolist()

    pathDf = pd.DataFrame(Parallel(n_jobs=mp.cpu_count(),
                                   backend="multiprocessing")
                          (delayed(bfs_encode)(traces, f) for f in tqdm(all_ids)))
    # one-hot
    lb_encoder = LabelEncoder()
    pathDf['pathCode'] = lb_encoder.fit_transform(pathDf['path'])

    pathDf.to_csv(f'data/{dataSet}/type.csv')
