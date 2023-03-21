import numpy as np


def add_sparse_anomaly(X, an_ratio=0.1, type='constant', mag=1):
    """ Adds sparse constants to the data given with the cardinality
    ratio of the anomaly equal to an_ratio 

    Args:
        X (np.ndarray): Ground truth data.
        an_ratio (float, optional): Ratio of the anomaly support/data size.
            Defaults to 0.1.
        type (str, optional): Anomaly type. Defaults to 'constant'.
        mag (int, optional): Magnitude of the anomaly. Defaults to 1.

    Returns:
        Xn (np.ndarray): Anomalous data
        anomaly_idx (np.ndarray): Indices of the anomalies
    """
    return Xn, anomaly_idx

def add_temporally_smooth_anomaly():
    """ Adds temporally smooth anomalies to the data with specified number
    of anomalous points, anomaly lengths and types.

    Args:
        X (np.ndarray): Ground truth data.
        NoAn (int, optional): Number of inserted anomalies. Defaults to X.size/20.
        LoAn (int, optional): Length of the inserted anomalies. Defaults to 5. 
        type (str, optional): Anomaly type. Defaults to 'constant'.
        mag (int, optional): Magnitude of the anomaly. Defaults to 1.

    Returns:
        Xn (np.ndarray): Anomalous data
        anomaly_idx (np.ndarray): Indices of the anomalies
    """
    pass

def add_locally_smooth_anomaly():
    """ Adds locally smooth anomalies to the data with specified number
    of anomalous points, anomaly lengths and types.

    Args:
        X (np.ndarray): Ground truth data.
        NoAn (int, optional): Number of inserted anomalies. Defaults to X.size/20.
        DoAn (int, optional): Diameter of the inserted anomalies. Defaults to 2. 
        type (str, optional): Anomaly type. Defaults to 'constant'.
        mag (int, optional): Magnitude of the anomaly. Defaults to 1.

    Returns:
        Xn (np.ndarray): Anomalous data
        anomaly_idx (np.ndarray): Indices of the anomalies
    """
    pass
