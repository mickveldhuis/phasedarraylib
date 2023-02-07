import numpy as np
import scipy.special as sp
import scipy.signal.windows as win


def symmetric_indices(count):
    n = np.arange(count)
    n_tilde = n - (count - 1) / 2

    return n_tilde


def seperable_weights(weights):
    w_mn = np.outer(weights, weights)
    
    return w_mn.flatten()


def cosine(element_count):
    sine_term = np.sin(np.pi / (2 * element_count))

    n_tilde = symmetric_indices(element_count)
    cosine_term = np.cos(np.pi / element_count * n_tilde)
    
    return sine_term * cosine_term


def raised_cosine(element_count, p=0):
    sine_term = np.sin(np.pi/(2*element_count))
    c_p = p/element_count + (1-p) * sine_term

    n_tilde = symmetric_indices(element_count)
    cosine_term = np.cos(np.pi/element_count * n_tilde)
    
    return c_p * (p + (1-p) * cosine_term)


def cosine_m(element_count, m=2, norm=True):
    n_tilde = symmetric_indices(element_count)

    cosine_term = np.cos(np.pi/element_count * n_tilde)
    
    window = np.power(cosine_term, m)

    if norm:
        window /= window.max()

    return window


def raised_cosine_squared(element_count, p=0, norm=True):
    n_tilde = symmetric_indices(element_count)

    cosine_term = np.cos(2*np.pi*n_tilde/element_count)

    window = c_p/2 * ((1+p) + (1-p)*cosine_term)

    if norm:
        window /= window.max()

    return window


def hamming(element_count):
    n_tilde = symmetric_indices(element_count)

    return 0.54 + 0.46*np.cos(2*np.pi*n_tilde/element_count)


def blackman_harris(element_count):
    n_tilde = symmetric_indices(element_count)

    return 0.42 + 0.5*np.cos(2*np.pi*n_tilde/element_count) + 0.08*np.cos(4*np.pi*n_tilde/element_count)


def kaiser(element_count, beta=5, norm=True):
    n_tilde = symmetric_indices(element_count)
    n_scaled = np.sqrt(1-np.power(2*n_tilde/element_count, 2))
    
    window = sp.i0(beta * n_scaled)

    if norm:
        window /= window.max()

    return window


def chebyshev(element_count, attenuation=40):
    return win.chebwin(element_count, attenuation)


def taylor(element_count, nbar=4, sll=30):
    return win.taylor(element_count, nbar=nbar, sll=sll)