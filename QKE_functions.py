from traceback import print_tb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale
import numpy as np

def reduceAttributeDimensions(n_dim, sample_train, sample_test):
    # Reduce dimensions
    pca = PCA(n_components=n_dim).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    # Normalise
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    # Scale
    digits_samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(digits_samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    return sample_train, sample_test
def reduceClassDimensions(X_raw, Y_raw, n_class):
    X = []
    Y = []

    cnt = 0
    for x, y in zip(X_raw, Y_raw):
        if y < n_class:
            X.append(np.array(x))
            Y.append(y)

            cnt = cnt + 1
            if cnt == 500:
                break
    return X, Y