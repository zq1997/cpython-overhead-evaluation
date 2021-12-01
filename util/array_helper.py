import numpy as np


def sort_arrays(values, *arrays):
    argsort = values.argsort()[::-1]
    arrays = iter(arrays)
    for arr in arrays:
        if arr is None:
            break
        arr[:] = arr[argsort]
    return map(lambda arr: arr[argsort], arrays)


def summarize(arr, fmt):
    return '(Q0 to Q4) %s    (mean) %s    (std) %s' % (
        ' '.join(fmt % q for q in np.quantile(arr, (0, 0.25, 0.5, 0.75, 1))),
        fmt % np.mean(arr),
        fmt % np.std(arr)
    )


def remap_dist_by_arr(dist, remap_arr, size):
    new_shape = list(dist.shape)
    new_shape[-1] = size
    new_dist = np.zeros(new_shape, dist.dtype)
    for i in range(size):
        new_dist[..., i] = dist[..., remap_arr == i].sum(-1)
    return new_dist


def normalize(dist, axis=1):
    return dist / dist.sum(axis, keepdims=True)
