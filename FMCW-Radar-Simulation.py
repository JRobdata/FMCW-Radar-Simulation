import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

###simulation parameters

f_c = 77e9      # carrier frequency (Hz)
B  = 4e9        # bandwidth (Hz)
Tc  = 1e-6      # duration of a chirp (Seconds)
A  = 10         # Amplitude (units)
Tg = 1e-6       # guard time between chirps (Seconds)
num_chirps = 100
v = 30          # velocity of the target (m/s)
RangeMax = 200  # Theoretical range limit of radar (metres)
R0  = 100       # initial target distance (metres)

###Constants
S = B/Tc
Tcycle = Tc + Tg
cycle_ints = np.arange(num_chirps)    # chirp indices
Ranges = R0 + v * cycle_ints * Tcycle # range at the start of each chirp
Taus = 2 * Ranges / c                 # time taken for the chirp to reach an object and back
wavelength = c / f_c                  # carrier wavelength (metres)
alpha = 1/Ranges**2                   # attenuation

#Nyquist-Shannon sampling theorem: sampling frequency f_s chosen to be at least 2f_max
f_b_max = 2*S*RangeMax/c  # max beat frequency
f_s = 2 * f_b_max 
dt = 1 / f_s              # sample spacing (Seconds)
###

t = np.arange(0, Tc, dt)  # fast-time samples within each chirp

std_deviation = 0.001
##########################

#Functions:
def generate_if_signal(Taus, alpha, t, f_c, S, A, std_deviation):    
    """Generates the noisy dechirped IF signal."""

    alpha_col = alpha[:,None]
    Taus_col = Taus[:,None]
    t_row = t[None,:] 

    IF_array_clean =  A * alpha_col * np.exp(1j*(2*np.pi*f_c*Taus_col + 2*np.pi* S*Taus_col*t_row - np.pi*S*Taus_col**2))
    gaussian_noise = (np.random.normal(0, std_deviation, IF_array_clean.shape) + 1j * np.random.normal(0, std_deviation, IF_array_clean.shape))

    return IF_array_clean + gaussian_noise   


def estimate_range_velocity(IF_array, dt, Tcycle, S, wavelength):
    """Estimates the target range and velocity from IF data."""
    
    IF_fft = np.fft.fft(IF_array, axis=1) #fft for each chirp
    mags = np.abs(IF_fft)
    fft_freqs = np.fft.fftfreq(IF_array.shape[1], d=dt)

    peak_index = np.argmax(mags[0])
    f_b = fft_freqs[peak_index]

    measured_range = c * f_b / (2*S)    #(metres)

    
    slowwave = IF_fft[:, peak_index]    # selected range bin across all chirps
    slowwave_fft = np.fft.fft(slowwave) # slow-time FFT for Doppler estimation
    slow_mag = np.abs(slowwave_fft)

    doppler_freqs = np.fft.fftfreq(IF_fft.shape[0], d=Tcycle)    
    doppler_freqs_shift = np.fft.fftshift(doppler_freqs, axes=None)
    slow_mag_shift = np.fft.fftshift(slow_mag)    
    doppler_index = np.argmax(slow_mag_shift)
    f_d = doppler_freqs_shift[doppler_index]   #doppler frequency

    measured_velocity = wavelength * f_d / 2   # (m/s)

    return measured_range, measured_velocity, doppler_freqs_shift, slow_mag_shift


def results_presentation(doppler_freqs_shift, slow_mag_shift, measured_range, R0, measured_v, v):
    """Plots Doppler results and compares the measured and simulated target parameters."""
    
    plt.figure()
    plt.plot(doppler_freqs_shift, slow_mag_shift)
    plt.xlabel("Doppler frequency (Hz)")
    plt.ylabel("FFT magnitude")
    plt.title("Doppler Spectrum")
    
    plt.show()
    
    print(f"Measured range = {measured_range:.2f}m")
    print(f"Actual range = {R0:.2f}m")
    print()
    print(f"Measured velocity = {measured_v:.2f}m/s")
    print(f"Actual velocity = {v:.2f}m/s")
    
##########################

def main():
    
    IF_array = generate_if_signal(Taus, alpha, t, f_c, S, A, std_deviation)
    measured_range, measured_v, doppler_freqs_shift, slow_mag_shift = (estimate_range_velocity(IF_array, dt, Tcycle, S, wavelength))
    results_presentation(doppler_freqs_shift, slow_mag_shift, measured_range, R0, measured_v, v)

##########################

if __name__ == "__main__":
    main()
