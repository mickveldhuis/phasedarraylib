import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from phasedarraylib.utilities import rotate, plot_configuration, plot_array_factor_1d, plot_array_factor_2d, arg_closest, frequency_to_wavelength
from phasedarraylib.geometry import linear, rectangular, random_rectangle, random_circle

class Array():
    """Class representing an arbitrary antenna array."""
    def __init__(self):
        """"The Array class constructor.

        Parameters
        ----------
        
        ... (type): ...
        """
        self.element_count = None
        self.element_positions = None
        self.weights = None

        self.steering_direction = None
        self.steering_phase = None
        
        self.data = None
        
        # Add a vectorized instance of the array_factor function
        self._array_factor = np.vectorize(self.array_factor)
    
    def set_element_positions(self, positions):
        positions = np.array(positions)

        if positions.ndim == 2 and positions.shape[1] != 2:
            raise ValueError('Passed array is not of the right shape. Expects (N, 2), got {}'.format(positions.shape))

        self.element_positions = positions

    def set_orientation(self, angle, center=np.array([0, 0]), show_plot=False):
        self.element_positions = rotate(self.element_positions, center, angle)
        
        if show_plot:
            self.plot_configuration()
    
    def set_offset(self, offset, show_plot=False):
        self.element_positions += offset
        
        if show_plot:
            self.plot_configuration()
    
    def steer_towards(self, u, v):
        self.steering_direction = np.array([u, v])

        if u == 0 and v == 0:
            self.steering_direction = None

    def set_weights(self, weights):
        self.weights = weights

    def array_factor(self, u, v, freq):
        """"Returns the array vector given direction cosines u, v.

        Parameters
        ----------
        u: float 
            direction cosine (x)
        v: float
            direction cosine (y)
        freq: float
            frequency in MHz

        Returns
        -------
        af: float
            returns the dot product of the antenna weights and 
        """        
        uv = np.array([u, v])
        lam = frequency_to_wavelength(freq)
        exp_term = np.exp(1j*2*np.pi/lam * (uv @ self.element_positions.T))
        
        af = np.dot(self.weights, exp_term)

        if self.steering_direction is not None:
            steering_phase = np.exp(-1j*2*np.pi/lam * (self.steering_direction @ self.element_positions.T))
            exp_steered_term = np.multiply(steering_phase.T, exp_term)

            af = np.dot(self.weights, exp_steered_term)
        
        return af

    def evaluate_array_factor(self, freq, spacing=0.01, sample_angles=False):
        """"Returns the array vector.

        Parameters
        ----------
        freq: float or sequence of floats
            frequency in MHz
        spacing: float
            spacing in the direction cosine or angles
        sample_angles: bool
            if False sample in uv coordinates otherwise 
            sample in degrees

        Returns
        -------
        af: float
            the array factor
        u_vals: float
            the direction cosine sampled in the x-direction
        v_vals: float
            the direction cosine sampled in the y-direction
        """
        uv_vals = np.arange(-1, 1 + spacing, spacing)

        angles = np.arange(180, 360 + spacing, spacing)
        angles_rad = np.radians(angles)

        u_vals = uv_vals if not sample_angles else np.cos(angles_rad)
        v_vals = uv_vals if not sample_angles else np.cos(angles_rad)
        
        us, vs = np.meshgrid(u_vals, v_vals)
        
        if not isinstance(freq, list | tuple | np.ndarray):
            af = self._array_factor(us, vs, freq)
            self.data = dict(
                af=af, 
                u=u_vals, 
                v=v_vals, 
                angles=angles, 
                freq=freq
            )

            return self.data
        
        af = []
        frequencies = []
        
        for f in freq:
            af.append(self._array_factor(us, vs, f))
            frequencies.append(f)
           
        self.data = dict(
            af=np.array(af), 
            u=u_vals, 
            v=v_vals, 
            angles=angles, 
            freq=np.array(frequencies)
        )
        
        return self.data
        
    def plot_1d(self, uv_coordinate=0, axis='u', freq_idx=0, db_threshold=-40, norm=True):
        if not self.data:
            raise ValueError('The Array Factor has not yet been evaluated!')

        af_2d = self.data.get('af')
        freq = self.data.get('freq')
        uv_vals = self.data.get('u')

        if isinstance(freq, list | tuple | np.ndarray):
            af_2d = af_2d[freq_idx]
            freq = freq[freq_idx]
        
        # Select slice of the array factor
        uv_idx = arg_closest(uv_vals, uv_coordinate)
        af = af_2d[uv_idx, :]

        if axis == 'v' or axis == 1:
            af = af_2d[:, uv_idx]
        
        if norm:
            af /= af.max()

        plot_array_factor_1d(af, uv_vals, freq, uv=uv_coordinate, axis=axis, db_threshold=db_threshold)
    
    def plot_2d(self, xframes=3, fn='', db_threshold=-40, norm=True):
        if not self.data:
            raise ValueError('The Array Factor has not yet been evaluated!')
        
        af = self.data.get('af')
        freq = self.data.get('freq')
        
        if norm:
            for i in range(freq.size):
                af[i] = af[i] / af[i].max()
        
        plot_array_factor_2d(af, freq, xframes=xframes, fn=fn, db_threshold=db_threshold)
    
    def plot_configuration(self):
        ant_x = self.element_positions[:, 0]
        ant_y = self.element_positions[:, 1]

        plot_configuration(ant_x, ant_y, aspect='auto')


class UniformLinearArray(Array):
    """Class representing a planar antenna array."""
    def __init__(self, length, spacing):
        super().__init__()
        
        self.element_count = length
    
        self.element_spacing = spacing
        self.element_positions = linear(length, spacing)
    
        self.weights = np.ones(length) / length
    
    def analytical_array_factor(self, u, freq, scan_u=0):
        lam = frequency_to_wavelength(freq)
        
        x = np.pi/lam * self.element_spacing * u

        if scan_u != 0:
            x -= np.pi/lam * self.element_spacing * u_scan

        af = np.sin(self.element_count * x)/np.sin(x)
        
        return af/self.element_count
    
    def analytical_nulls(self, freq, m=1):
        lam = frequency_to_wavelength(freq)

        return m * lam / (self.element_count * self.element_spacing)
    
    def nullnullbeamwidth(self, freq, deg=False):
        u_0 = self.analytical_nulls(freq)

        bw = 2*np.arcsin(u_0)
        
        if deg:
            bw = np.degrees(bw)

        return bw, u_0

    def halfpowerbeamwidth(self, freq, deg):
        def root_function(x):
            return np.power(self.analytical_array_factor(x, freq), 2) - 0.5

        u = opt.brentq(root_function, 1e-10, 1)
        
        hpbw_u = 2*u
        hpbw_theta = 2*np.arcsin(u)

        if deg:
            hpbw_theta = np.degrees(hpbw_theta)

        return hpbw_theta, hpbw_u 


class PlanarArray(Array):
    """Class representing a planar antenna array."""
    def __init__(self, shape=None, spacing=None):
        super().__init__()
        
        if shape is not None or spacing is not None:
            self.nx = shape[0]
            self.ny = shape[1]
            self.element_count = self.nx * self.ny
        
            self.element_spacing = spacing
            self.element_positions = rectangular(*shape, spacing)
        
            self.weights = np.ones(self.element_count) / self.element_count
    
    def from_arrays(self, arrays):
        positions = None
        
        for array in arrays:
            if positions is None:
                positions = array.element_positions 
                
                continue

            positions = np.concatenate([positions, array.element_positions])

        self.element_positions = np.array(positions)
        self.element_count = self.element_positions.shape[0]
        self.weights = self.weights = np.ones(self.element_count) / self.element_count


class RandomRectangularArray(Array):
    """Class representing a planar antenna array."""
    def __init__(self, size, extent):
        super().__init__()
        
        self.element_count = size

        self.element_positions = random_rectangle(size, extent=extent)
    
        self.weights = np.ones(size) / size


class RandomCircularArray(Array):
    """Class representing a planar antenna array."""
    def __init__(self, size, radial_extent):
        super().__init__()
        
        self.element_count = size

        self.element_positions = random_circle(size, radial_extent)
    
        self.weights = np.ones(size) / size