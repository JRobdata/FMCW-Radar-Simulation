# FMCW-Radar-Simulation

Detects the range and radial velocity of multiple target using a simulated frequency modulated continuous wave (FMCW) radar. 

---

## Features

- Simulates an FMCW radar system.
- Generates a complex baseband intermediate-frequency (IF) signal by mixing the transmitted chirps as they are sent, and the  received chirps as they are received from the moving target.
- Adds complex Gaussian noise, and attenuation.
- Includes Hann window process to prevent spectral leakage.
- Uses fast-time FFT for range estimation.
- Uses slow-time FFT for Doppler velocity estimation.
- Detects target range peaks using SciPy peak detection.
- Plots the Doppler frequency spectrum.


## Dependencies

This script uses:

- NumPy
- SciPy
- Matplotlib

To install using `conda`:

```bash
conda install numpy matplotlib scipy
```

## Example Output

![Doppler spectrum](doppler_spectrum.png)

## Notes

This is a simplified model with multiple targets intended for exploration of FMCW radar signal processing rather than to reproduce a fully realistic radar system. Performance may degrade when long-range targets are detected alongside short-range ones.

Further features may be implemented such as, angular estimation, range Doppler mapping, CFAR, and animation .


