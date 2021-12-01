import matplotlib.pyplot
import numpy as np


def hsy_to_rgb(h, s, y, *, as_type=None):
    rgb_weight = np.asarray([0.2126, 0.7152, 0.0722])

    h, s, y = np.broadcast_arrays(h, s, y)
    input_shape = h.shape

    h = 2. * np.pi * h[..., None]
    s = s[..., None]
    y = y[..., None]

    mat = np.empty([*input_shape, 3, 3], dtype=np.float)
    shaft = np.asarray(rgb_weight)
    shaft = shaft / np.linalg.norm(shaft)
    normal = np.cross([1, 0, 0], shaft)
    mat[..., 0, :] = rgb_weight
    mat[..., 1, :] = np.dot(shaft, normal) * shaft + \
                     np.cos(h) * np.cross(np.cross(shaft, normal), shaft) + \
                     np.sin(h) * np.cross(shaft, normal)

    vec = np.empty([*input_shape, 3], dtype=np.float)
    vec[..., :2] = mat[..., :2, :].sum(-1) * y

    check_mat = np.cross(mat[..., 1, :], shaft)
    check_vec = check_mat.sum(-1, keepdims=True) * y

    candidates = np.empty([*input_shape, 6, 3])
    for plane in (0, 1, 2):
        mat[..., 2, :] = 0
        mat[..., 2, plane] = 1.
        vec[..., 2] = 0.
        candidates[..., plane, :] = np.linalg.solve(mat, vec)
        vec[..., 2] = 1.
        candidates[..., plane + 3, :] = np.linalg.solve(mat, vec)

    c_min = np.min(candidates, -1)
    c_max = np.max(candidates, -1)
    valid = np.isclose(c_min, 0) | (c_min > 0)
    valid &= np.isclose(c_max, 1) | (c_max < 1)
    diff = (check_mat[..., None, :] * candidates).sum(-1) - check_vec
    valid &= np.isclose(diff, 0) | (diff < 0)
    assert valid.any(-1).all()

    candidates = candidates.reshape([-1, 6, 3])
    valid = valid.reshape([-1, 6])
    rgb = candidates[np.arange(len(valid)), valid.argmax(-1)].reshape([*input_shape, 3])
    rgb = y + (rgb - y) * s

    assert np.allclose(np.dot(rgb, rgb_weight), y[..., 0])

    if as_type is float:
        return rgb.clip(0, 1)
    else:
        rgb = np.clip((rgb * 256).astype(np.int), 0, 255)
        if as_type is int:
            return rgb
        else:
            rgb = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
            return np.vectorize(lambda x: '#%06x' % x)(rgb)


if __name__ == '__main__':
    fig, axs = matplotlib.pyplot.subplots(4, 3, figsize=[8.3, 11.7])
    for i, ax in enumerate(axs.flatten()):
        width = 100
        height = 100
        H = np.arange(width).reshape([1, width]).repeat(height, 0) / (width - 1)
        S = np.arange(height).reshape([height, 1]).repeat(width, 1) / (height - 1)
        Y = (i + 0.5) / axs.size
        ax.set_title('Y = %.2f' % Y)
        img = hsy_to_rgb(H, S, Y, as_type=float)
        ax.imshow(img, origin='lower')
        ax.set_xticks(np.arange(6) * (width - 1) / 5)
        ax.set_xticklabels(['%.1f' % x for x in ax.get_xticks() / (width - 1)])
        ax.set_yticks(np.arange(6) * (height - 1) / 5)
        ax.set_yticklabels(['%.1f' % x for x in ax.get_yticks() / (height - 1)])
        ax.xaxis.tick_top()
    fig.tight_layout()
    fig.savefig('hsy_color.pdf')
