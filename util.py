import numpy as np


def otsu_threshold(gray):
    h, w = gray.shape
    count = {i: 0 for i in range(256)}
    for i in range(h):
        for j in range(w):
            count[gray[i, j]] += 1
    prob = np.array([count[i] / float(h * w) for i in sorted(count)])
    means = np.array([prob[i] * (i + 1) for i in count])
    mean = np.sum(means)
    minvar = -np.inf
    minT = 0
    for t in range(256):
        w1 = np.sum([i for i in prob[:t + 1]])
        w2 = 1.0 - w1
        if not w1 or not w2: continue
        m1 = np.sum([i for i in means[:t + 1]])
        mean1 = m1 / w1
        mean2 = (mean - m1) / w2
        bcvar = w1 * w2 * (mean2 - mean1) ** 2
        if bcvar > minvar:
            minvar = bcvar
            minT = t
    return minT