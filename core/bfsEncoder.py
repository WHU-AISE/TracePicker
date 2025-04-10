import networkx as nx
from core.pool import HistPool
from collections import defaultdict

from entity.Trace import Trace
import hashlib
from treelib import Tree, Node


class BFSEncoder:
    def __init__(self, poolHeight: int) -> None:
        self.pool = HistPool(height=poolHeight)
        self.bufferLabels = []


    def buildTreeAndCheck(self,
                        trace: Trace):
        tree = Tree()
        # is anbnormal?
        isError, isPD = trace.isError, False
        # expected duration and true duration
        d_expected, d_true = 0, 0

        id2span = {}
        for span in trace.spans:
            id2span[span.getSpanId()] = span
        for span in trace.spans:
            label = span.getSpanLabel()
            # check performance degradation
            mu, std = self.pool.get_mu_std(label)
            d_expected += (mu + 5* std)
            d_true += span.duration
            self.pool.add(label, span.duration)
            self.bufferLabels.append(label)

            if tree.contains(span.spanId):
                continue

            if span.parentSpanId == '-1':
                tree.create_node(tag=label, identifier=span.spanId)
            else:
                tree.create_node(tag=label, identifier=span.spanId, parent=id2span[span.parentSpanId].spanId)
        

        isPD = d_true > d_expected
        return tree, (isError or isPD)


    def bfsEncode_tree(self, tree: Tree):
        path = str([tree.get_node(nid).tag for nid in tree.expand_tree(mode=Tree.WIDTH)])
        return hash(path)

    def getAllLabels(self):
        return self.pool.get_labels()

    def getBufferLabels(self):
        return list(set(self.bufferLabels))

    def clear(self):
        self.bufferLabels.clear()
