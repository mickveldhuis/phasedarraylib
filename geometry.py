import numpy as np


def rectangular(nx, ny, spacing):
    grid = np.array([(x, y) for x in range(nx) for y in range(ny)])

    return spacing * grid


def linear(n, spacing):
    line = np.array([(x, 0) for x in range(n)])

    return spacing * line


def random_rectangle(n, extent=(0, 1, 0, 1)):
    rng = np.random.default_rng(12345)

    x = rng.uniform(low=extent[0], high=extent[1], size=n)
    y = rng.uniform(low=extent[2], high=extent[3], size=n)

    return np.stack([x, y], axis=-1)

def random_circle(n, radius):
    radii = radius * np.sqrt(np.random.rand(n))
    angles = 2*np.pi * np.random.rand(n)

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    return np.stack([x, y], axis=-1)