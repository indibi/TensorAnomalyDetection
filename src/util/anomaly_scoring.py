import numpy as np

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from src.util.t2m import t2m
from src.util.m2t import m2t


def score_anomalies_ocsvm(s):
    # anomaly_scores = np.zeros(s.shape)
    s_d = t2m(s,4).T
    model = svm.OneClassSVM()
    model.fit(s_d)
    day_scores = model.score_samples(s_d)
    day_scores = day_scores.reshape((day_scores.size,1))

    return m2t(day_scores*s_d, s.shape, 4)

    
