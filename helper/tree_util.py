from treelib import Tree

from entity.Trace import Trace
def buildTree(trace: Trace):
        tree = Tree()
        id2span = {}
        for span in trace.spans:
            id2span[span.getSpanId()] = span
        for span in trace.spans:
            label = span.getSpanLabel()

            if tree.contains(span.spanId):
                continue

            if span.parentSpanId == '-1':
                tree.create_node(tag=label, identifier=span.spanId)
            else:
                tree.create_node(tag=label, identifier=span.spanId, parent=id2span[span.parentSpanId].spanId)
        
        return tree


def bfsEncode_tree(tree: Tree):
    path = str([tree.get_node(nid).tag for nid in tree.expand_tree(mode=Tree.WIDTH)])
    return hash(path)