from trace import Trace
from itertools import chain
from collections import defaultdict

class SharedBuffer:
    def __init__(self) -> None:
        self.map = defaultdict(list) # {code, traces}
        self.count = 0
    

    def __len__(self):
        return self.count
    
    
    def add(self, 
            code: str, 
            trace: Trace,
            isAb: bool):
        trace.abnormal = isAb
        self.map[code].append(trace)
        self.count += 1
    
    
    def clear(self):
        self.map = defaultdict(list)
        self.count = 0

    
    def getCodes(self):
        return list(self.map.keys())
    

    def getTracesByCode(self, code):
        if code is None:
            return list(chain(*self.map.values()))
        else:
            return self.map[code]
    

    def countByCode(self, code):
        return len(self.map[code])
    

    def remove(self, code, trace):
        self.map[code].remove(trace)
        if len(self.map[code]) == 0:
            del self.map[code]
