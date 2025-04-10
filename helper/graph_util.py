import networkx as nx

from entity.Trace import Trace, Span
import hashlib

def build_dg_with_trace(trace: Trace, level: str):
     # init a DAG
    dg = nx.DiGraph()
    edges=[]
    for span in trace.spans:
        label = span.service if level == 'svc' else span.getSpanLabel()
        if span.parentSpanId == '-1':
            edges.append(('ROOT', label))
        else:
            # find parent
            parentSpan = None
            for s in trace.spans:
                if s.spanId == span.parentSpanId:
                    parentSpan = s
            if parentSpan == None:
                continue
            parentLabel = parentSpan.service if level == 'svc' else parentSpan.getSpanLabel()
            edges.append((parentLabel, label))
    edges = sorted(edges)
    dg.add_edges_from(edges)
    return dg


def get_durations(trace: Trace):
    # all span labels
    labels = list(set([span.getSpanLabel() for span in trace.spans]))
    labels.sort()
    # duration array
    durations = [0]*len(labels)
    for span in trace.spans:
        label = span.getSpanLabel()
        durations[labels.index(label)]=span.duration
    return durations


def get_root(dg: nx.DiGraph):
    dg.remove_edges_from(nx.selfloop_edges(dg))
    for node in dg.nodes:
        if dg.in_degree(node) == 0:
            return node


def get_leaves(dg: nx.DiGraph):
    leaves = []
    for node in dg.nodes:
        if dg.out_degree(node) == 0:
            leaves.append(node)
    return leaves


def bfs_encode(dg: nx.DiGraph):
    root = get_root(dg)
    path = str(list(nx.bfs_tree(dg, root)))
    return hash(path)



def GMTA_hash(trace: Trace):
    '''
        Xiaofeng Guo et.al. 
        Graph-based trace analysis for microservice architecture understanding and problem diagnosis.
        FSE 2020
    '''
    def span_hash(span: Span, offset: int):
        span_code = hash(span.service) + hash(span.operation) + (offset * 2)

        for child in trace.getSubSpans(span.spanId):
            span_code += span_hash(child, offset+1)

        return span_code

    root = trace.getRoot()
    return span_hash(root, 0)
