from collections import deque
import json
import copy

class Span:
    def __init__(self, startTime, duration, statusCode, traceId, spanId, parentSpanId, instance, service, operation):
        self.startTime = startTime
        self.duration = duration
        self.statusCode = statusCode
        self.traceId = traceId
        self.spanId = spanId
        self.parentSpanId = parentSpanId
        self.instance = str(instance)
        self.service = str(service)
        self.operation = str(operation)
    
    def getElapsedTime(self):
        return self.duration
    
    def getTraceId(self):
        return self.traceId
    
    def getSpanId(self):
        return self.spanId
    
    def getParentId(self):
        return self.parentSpanId
    
    def getSpanLabel(self):
        return str(self.service) + ':' + str(self.operation)
    
    def buildSpan(record):
        startTime = record['startTime']
        duration = record['duration']
        statusCode = record['statusCode']
        traceId = record['traceId']
        spanId = record['spanId']
        parentSpanId = record['parentSpanId']
        service = record['service']
        instance = record['cmdb_id']
        operation = record['operation']
        return Span(startTime, duration, statusCode, traceId, spanId, parentSpanId, instance, service, operation)
    
    def serialize(span):
        return json.dumps(span.__dict__)
    
    def deserialize(spanStr):
        spanDict = json.loads(spanStr)
        return Span(spanDict['startTime'], 
                    spanDict['duration'], 
                    spanDict['statusCode'], 
                    spanDict['traceId'], 
                    spanDict['spanId'], 
                    spanDict['parentSpanId'],
                    spanDict['instance'],
                    spanDict['service'],
                    spanDict['operation'])
    
    def toString(self):
        return ("traceId: " + str(self.traceId) + ",\n" +
                "spanId: " + str(self.spanId) + ",\n" +
                "parentSpanId: " + str(self.parentSpanId) + ",\n" +
                "duration: " + str(self.duration) + ",\n" +
                "statusCode: " + str(self.statusCode) + ",\n" +
                "service: " + str(self.service) + ",\n" +
                "operation: " + str(self.operation))
        
class Trace:
    def __init__(self, traceID, spans, isError=False):
        self.traceID = traceID
        self.spans = spans
        self.isError = isError

        # strainer
        self.abnormal = False
        self.durations = []

    def getTraceID(self):
        return self.traceID
    
    def getSpanNum(self):
        return len(self.spans)
    
    def getSpans(self):
        return self.spans

    def getRoot(self) -> Span:
        return [span for span in self.spans if span.getParentId()=='-1'][0]
    
    def getSpansWithDepth(self):
        root = self.getRoot()
        res = []
        queue = deque([(root, 0)])
        while queue:
            current_node, current_depth = queue.popleft()
            res.append((current_node, current_depth))
            for child in self.getSubSpans(current_node.spanId):
                queue.append((child, current_depth + 1))
        return res
    
    def getSubSpans(self, spanId: str):
        childs = []
        for span in self.spans:
            if span.getParentId() == spanId:
                childs.append(span)
        return childs
        
    def serialize(trace):
        traceDict = copy.deepcopy(trace.__dict__)
        spansDict = []
        for sp in trace.spans:
            spansDict.append(Span.serialize(sp))
        traceDict['spans'] = spansDict
        return json.dumps(traceDict)
    
    def deserialize(traceStr):
        traceDict = json.loads(traceStr)
        spans = [Span.deserialize(sp) for sp in traceDict['spans']]
        return Trace(traceDict['traceID'],
                     spans, traceDict.get('isError', False))
