import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.signal import find_peaks

###simulation parameters

f_c = 77e9      # carrier frequency (Hz)
B  = 4e9        # bandwidth (Hz)
Tc  = 1e-6      # duration of a chirp (Seconds)
A  = 10         # Amplitude (units)
Tg = 1e-6       # guard time between chirps (Seconds)
num_chirps = 100
std_deviation = 0.001 # for gaussian noise

#velocites of the targets (m/s)
velocities = np.array([30, -70, 2, 5]) #velocites of the targets (m/s)
ranges_init = np.array ([100, 50, 90, 10 ]) #ranges of the targets (metres)

RangeMax = 200  # Theoretical range limit of radar 

###Constants
S = B/Tc
Tcycle = Tc + Tg
cycle_ints = np.arange(num_chirps)    # chirp indices
Ranges = ranges_init[:,None] + velocities[:,None] * cycle_ints[None,:] * Tcycle # range at the start of each chirp
Taus = 2 * Ranges / c  # time taken for the chirp to reach an object and back(seconds)    
wavelength = c / f_c                  # carrier wavelength (metres)
alpha = 1/Ranges**2                   # attenuation


#Nyquist-Shannon sampling theorem: sampling frequency f_s chosen to be at least 2f_max
f_b_max = 2*S*RangeMax/c  # max beat frequency
f_s = 2 * f_b_max 
dt = 1 / f_s              # sample spacing (Seconds)
###

t = np.arange(0, Tc, dt)  # fast-time samples within each chirp

###resolutions
Tf = num_chirps * Tcycle #total frame time
vmax = wavelength/(4 * Tcycle)
vres = wavelength/(2 * Tf)
Rangeres = c/(2 * B)


print(f"Max velocity measurement: {vmax:.2f} m/s")
print(f"Velocity resolution: {vres:.2f} m/s")
print(f"Range resolution: {Rangeres:.3f} m")
print()      

##########################

#Functions:
     

def generate_if_signal(Taus, alpha, t, f_c, S, A):    
    """Generates the dechirped IF signal."""

    alpha_3d = alpha[:, :, None]
    Taus_3d = Taus[:, :, None]
    t_3d = t[None, None, :]

    IF_array_clean =  A * alpha_3d * np.exp(1j*(2*np.pi*f_c*Taus_3d + 2*np.pi* S*Taus_3d*t_3d - np.pi*S*Taus_3d**2))
    IF_array_sum_clean = np.sum(IF_array_clean, axis=0)
    
    return IF_array_sum_clean

def add_noise(IF_array, std_deviation):
    """Add noise to the IF signal"""

    gaussian_noise = (np.random.normal(0, std_deviation, IF_array.shape) + 1j * np.random.normal(0, std_deviation, IF_array.shape))
    
    return IF_array + gaussian_noise

def estimate_range_velocity(IF_array_sum, dt, Tcycle, S, wavelength):
    """Estimates the target range and velocity from IF data."""
    
    hanning_window = np.hanning(IF_array_sum.shape[1])
    IF_windowed = IF_array_sum * hanning_window[None, : ]
    
    IF_fft = np.fft.fft(IF_windowed, axis=1) #fft for each chirp
    mags_per_chirp = np.abs(IF_fft)
    fft_freqs = np.fft.fftfreq(IF_array_sum.shape[1], d=dt)


    peak_indices, _ = find_peaks(x=mags_per_chirp[0], prominence=(np.max(mags_per_chirp[0]) - np.min(mags_per_chirp[0])) * 1e-6)   #Only first chirp peaks are necessary
  

    f_bs = fft_freqs[peak_indices]

    measured_ranges = c * f_bs / (2*S)    #(metres)

    
    slowwave = IF_fft[:, peak_indices]    # selected range bin across all chirps
    slowwave_fft = np.fft.fft(slowwave, axis=0) # slow-time FFT for Doppler estimation
    slow_mag = np.abs(slowwave_fft)

    doppler_freqs = np.fft.fftfreq(IF_fft.shape[0], d=Tcycle)    
    doppler_freqs_shift = np.fft.fftshift(doppler_freqs)
    slow_mag_shift = np.fft.fftshift(slow_mag,axes=0)    
    doppler_indices = np.argmax(slow_mag_shift, axis=0)
    f_ds = doppler_freqs_shift[doppler_indices]   #doppler frequency

    measured_velocities = wavelength * f_ds / 2   # (m/s)

    return measured_ranges, measured_velocities, doppler_freqs_shift, slow_mag_shift


def results_presentation(doppler_freqs_shift, slow_mag_shift, measured_ranges, ranges_init, measured_velocities, velocities):
    """Plots Doppler results and compares the measured and simulated target parameters."""
    
    plt.figure()
    plt.plot(doppler_freqs_shift, slow_mag_shift)
    plt.xlabel("Doppler frequency (Hz)")
    plt.ylabel("FFT magnitude")
    plt.title("Doppler Spectrum")
    
    plt.show()
    
    order = np.argsort(ranges_init)
    ranges_init_sorted = ranges_init[order]
    velocities_sorted = velocities[order]
    
    for r, v, r_actual, v_actual in zip(measured_ranges, measured_velocities, ranges_init_sorted, velocities_sorted):
        print(f"Measured: range = {r:.2f} m, velocity = {v:.2f} m/s / Actual: range = {r_actual:.2f} m, velocity = {v_actual:.2f} m/s")
    
##########################

def main():
        
    IF_array_sum_clean = generate_if_signal(Taus, alpha, t, f_c, S, A)
    IF_array_sum = add_noise(IF_array_sum_clean, std_deviation)
    
    measured_ranges, measured_velocities, doppler_freqs_shift, slow_mag_shift = (estimate_range_velocity(IF_array_sum, dt, Tcycle, S, wavelength))
    results_presentation(doppler_freqs_shift, slow_mag_shift, measured_ranges, ranges_init, measured_velocities, velocities)

##########################

if __name__ == "__main__":
    main()
