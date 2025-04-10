from collections import deque

import numpy as np

class HistPool:
    def __init__(self, height):
        self.limit = height
        self.data = {}
        self.db = {}

        self.count = 0
        self.threshold = 100
        

    def __len__(self):
        return len(self.data)
    
                  
    def add(self, 
            label: str,
            duration: float):
        if label not in self.data.keys():
            self.data[label] = deque(maxlen=self.limit)
        self.data[label].append(duration)

        self.count += 1
        if self.count == self.threshold:
            # scheduled updates
            self.cal_mu_std()
            self.count=0
            self.threshold = min(self.threshold + 100, 2000)
            

    def cal_mu_std(self):
        for label in self.data.keys():
            mu = np.mean(self.data[label])
            std = np.std(self.data[label])
            self.db[label] = (mu, std)
        

    def get_mu_std(self, label):
        if label not in self.db.keys():
            return 0, 0
        else:
            mu, std = self.db[label]
            return mu, std
        
    def get_labels(self):
        return self.data.keys()
# from collections import Counter, deque

# import numpy as np

# class HistPool:
#     def __init__(self, height):
#         self.limit = height
#         self.data = {}
#         self.db = {}

#         self.updateThresholds = {}
#         self.labelCounter = Counter()
        

#     def __len__(self):
#         return len(self.data)
    
                  
#     def add(self, 
#             label: str,
#             duration: float):
#         if label not in self.data.keys():
#             self.data[label] = deque(maxlen=self.limit)
#             self.updateThresholds[label] = 10
#         self.data[label].append(duration)
#         self.labelCounter[label]+=1

#         if self.labelCounter[label] % self.updateThresholds[label] == 0:
#             self.cal_mu_std(label)
#             self.updateThresholds[label] = min(self.updateThresholds[label] * 10, 10000)
            

#     def cal_mu_std(self, label):
#         mu = np.mean(self.data[label])
#         std = np.std(self.data[label])
#         self.db[label] = (mu, std)
        

#     def get_mu_std(self, label):
#         if label not in self.db.keys():
#             return 0, 0
#         else:
#             mu, std = self.db[label]
#             return mu, std
        
#     def get_labels(self):
#         return self.data.keys()