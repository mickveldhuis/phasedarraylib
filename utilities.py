import numpy as np
import astropy.constants as const
import astropy.units as units


def direction_cosines(theta, phi):
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    w = np.cos(theta)
    
    return u, v, w


def decibel(x):
    return 20*np.log10(np.abs(x))


def frequency_to_wavelength(frequency):
    """
    Convert a frequency in MHz to a wavelength in meters.
    """
    return const.c.value/(frequency * units.MHz.to(1/units.s))


def rotate(points, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Parameters
    ----------
    points: ndarray (N, 2)
        vector containing the pionts to be rotated
    origin: ndarray  
        vector representing the origin of the rotation
    angle: float
        anti-clockwise rotation angle in radians

    Returns
    -------
    p_rotated: float
        returns the rotated points
    """
    R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
    
    o = np.atleast_2d(origin)
    p = np.atleast_2d(points)

    p_rotated = o.T + R @ (p.T - o.T)

    return np.squeeze(p_rotated.T)


def arg_closest(array, value):
    idx = np.abs(array - value).argmin()

    return idx