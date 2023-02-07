import numpy as np

from phasedarraylib.utilities import frequency_to_wavelength

class PhaseShiftBeamformer():
    def __init__(self, array, carrier_frequency, steering_angle):
        self.array = array
        self.frequency = carrier_frequency
        self.steering_angle = steering_angle
    
    def _phase_shifts(self, scheme, args=None):
        """"Calculate the phase shifts for all steering angles.

        Parameters
        ----------
        scheme: str
            name of the weighting function
        args: list/tuple
            list or tuple containing the function arguments

        Returns
        -------
        phase_shifts: (N, L) ndarray
            the element weights 
        """    

    def output(self, weighted_output=False):
        return
        
    
    def phases(self):
        lam = frequency_to_wavelength(self.frequency)
        
        return np.exp(1j*2*np.pi/lam * (uv @ self.element_positions.T))

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