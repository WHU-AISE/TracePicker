import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

def min_max_scaler(data):
    # (x - min) / (max - min)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data)
    return scaler

def quantile_scaler(data):
    scaler = QuantileTransformer()
    scaler.fit(data)
    return scaler

def standard_scaler(data):
    # (x-mean)/std
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler