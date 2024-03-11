#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#
from __future__ import annotations
import numpy as np
import math, cmath
from scipy import signal
from typing import Any, Tuple
#
# you should use the demodulator based on arctan given below as a reference
#
# in order to implement your OWN FM demodulator without the arctan function,
# a very good and to-the-point description is given by Richard Lyons at:
#
# https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/
#
# the demodulator boils down to implementing equation (13-117) from above, where
# the derivatives are nothing else but differences between consecutive samples
#
# needless to say, you should not jump directly to equation (13-117)
# rather try first to understand the entire thought process based on calculus
# identities, like derivative of the arctan function or derivatives of ratios
#

#
# use the four quadrant arctan function for phase detect between a pair of
# IQ samples; then unwrap the phase and take its derivative to demodulate
#

def custom_fm_demod(
    I: Any | np.ndarray[Any, np.dtype[Any]],
    Q: Any | np.ndarray[Any, np.dtype[Any]],
    prev_I: Any | np.ndarray[Any, np.dtype[Any]] = np.float64(0.0),
    prev_Q: Any | np.ndarray[Any, np.dtype[Any]] = np.float64(0.0),
) -> Tuple[
    np.ndarray[np.float64],
    Any | np.ndarray[Any, np.dtype[Any]],
    Any | np.ndarray[Any, np.dtype[Any]],
]:
    """Custom FM demodulator without using computationally heavy arctan function.
    params:
            I: in-phase samples
            Q: quadrature samples
            prev_I: previous in-phase sample
            prev_Q: previous quadrature sample
    returns:
            fm_demod: demodulated samples
            prev_I: last in-phase sample
            prev_Q: last quadrature sample
    """
    fm_demod: np.NDArray[np.float64] = np.empty(len(I))

    for k, _ in enumerate(I):

        denom = I[k] ** 2 + Q[k] ** 2
        # equation (13-117) from the link above - finite difference used as this is discretized
        if denom: 
            fm_demod[k] = ((I[k] * (Q[k] - prev_Q)) - (Q[k] * (I[k] - prev_I))) / denom
        else: 
            fm_demod[k] = np.float64(0.0)

        # save current I and Q to compute the next derivative
        prev_I = I[k]
        prev_Q = Q[k]

    return fm_demod, prev_I, prev_Q

def fmDemodArctan(I, Q, prev_phase = 0.0):
#
# the default prev_phase phase is assumed to be zero, however
# take note in block processing it must be explicitly controlled

	# empty vector to store the demodulated samples
	fm_demod = np.empty(len(I))

	# iterate through each of the I and Q pairs
	for k in range(len(I)):

		# use the atan2 function (four quadrant version) to detect angle between
		# the imaginary part (quadrature Q) and the real part (in-phase I)
		current_phase = math.atan2(Q[k], I[k])

		# we need to unwrap the angle obtained in radians through arctan2
		# to deal with the case when the change between consecutive angles
		# is greater than Pi radians (unwrap brings it back between -Pi to Pi)
		[prev_phase, current_phase] = np.unwrap([prev_phase, current_phase])

		# take the derivative of the phase
		fm_demod[k] = current_phase - prev_phase

		# save the state of the current phase
		# to compute the next derivative
		prev_phase = current_phase

	# return both the demodulated samples as well as the last phase
	# (the last phase is needed to enable continuity for block processing)
	return fm_demod, prev_phase

# custom function for DFT that can be used by the PSD estimate
def DFT(x):

	# number of samples
	N = len(x)

	# frequency bins
	Xf = np.zeros(N, dtype='complex')

	# iterate through all frequency bins/samples
	for m in range(N):
		for k in range(N):
			Xf[m] += x[k] * cmath.exp(1j * 2 * math.pi * ((-k) * m) / N)

	# return the vector that holds the frequency bins
	return Xf

# custom function to estimate PSD based on the Bartlett method
# this is less accurate than the Welch method from matplotlib
# however, as the visual inspections confirm, the estimate gives
# the user a "reasonably good" view of the power spectrum
def estimatePSD(samples, NFFT, Fs):

	# rename the NFFT argument (notation consistent with matplotlib.psd)
	# to freq_bins (i.e., frequency bins for which we compute the spectrum)
	freq_bins = NFFT
	# frequency increment (or resolution of the frequency bins)
	df = Fs/freq_bins

	# create the frequency vector to be used on the X axis
	# for plotting the PSD on the Y axis (only positive freq)
	freq = np.arange(0, Fs/2, df)

	# design the Hann window used to smoothen the discrete data in order
	# to reduce the spectral leakage after the Fourier transform
	hann = np.empty(freq_bins)
	for i in range(len(hann)):
		hann[i] = pow(math.sin(i*math.pi/freq_bins),2)

	# create an empty list where the PSD for each segment is computed
	psd_list = []

	# samples should be a multiple of frequency bins, so
	# the number of segments used for estimation is an integer
	# note: for this to work you must provide an argument for the
	# number of frequency bins not greater than the number of samples!
	no_segments = int(math.floor(len(samples)/float(freq_bins)))

	# iterate through all the segments
	for k in range(no_segments):

		# apply the hann window (using pointwise multiplication)
		# before computing the Fourier transform on a segment
		windowed_samples = samples[k*freq_bins:(k+1)*freq_bins] * hann

		# compute the Fourier transform using the built-in FFT from numpy
		Xf = np.fft.fft(windowed_samples, freq_bins)

		# note, you can check how MUCH slower is DFT vs FFT by replacing the
		# above function call with the one that is commented below
		#
		# Xf = DFT(windowed_samples)
		#
		# note: the slow impelementation of the Fourier transform is not as
		# critical when computing a static power spectra when troubleshooting
		#
		# note also: time permitting a custom FFT can be implemented

		# since input is real, we keep only the positive half of the spectrum
		# however, we will also add the signal energy of negative frequencies
		# to have a better a more accurate PSD estimate when plotting
		Xf = Xf[0:int(freq_bins/2)] # keep only positive freq bins
		psd_seg = (1/(Fs*freq_bins/2)) * (abs(Xf)**2) # compute signal power
		psd_seg = 2*psd_seg # add the energy from the negative freq bins

		# translate to the decibel (dB) scale
		for i in range(len(psd_seg)):
			psd_seg[i] = 10*math.log10(psd_seg[i])

		# append to the list where PSD for each segment is stored
		# in sequential order (first segment, followed by the second one, ...)
		psd_list.extend(psd_seg)

	# compute the estimate to be returned by the function through averaging
	psd_est = np.zeros(int(freq_bins/2))

	# iterate through all the frequency bins (positive freq only)
	# from all segments and average them (one bin at a time ...)
	for k in range(int(freq_bins/2)):
		# iterate through all the segments
		for l in range(no_segments):
			psd_est[k] += psd_list[k + l*int(freq_bins/2)]
		# compute the estimate for each bin
		psd_est[k] = psd_est[k] / no_segments

	# the frequency vector and PSD estimate
	return freq, psd_est

# custom function to format the plotting of the PSD
def fmPlotPSD(ax, samples, Fs, height, title):

	x_major_interval = (Fs/12)		# adjust grid lines as needed
	x_minor_interval = (Fs/12)/4
	y_major_interval = 20
	x_epsilon = 1e-3
	x_max = x_epsilon + Fs/2		# adjust x/y range as needed
	x_min = 0
	y_max = 10
	y_min = y_max-100*height
	ax.psd(samples, NFFT=512, Fs=Fs)
	#
	# below is the custom PSD estimate, which is based on the Bartlett method
	# it less accurate than the PSD from matplotlib, however it is sufficient
	# to help us visualize the power spectra on the acquired/filtered data
	#
	# freq, my_psd = estimatePSD(samples, NFFT=512, Fs=Fs)
	# ax.plot(freq, my_psd)
	#
	ax.set_xlim([x_min, x_max])
	ax.set_ylim([y_min, y_max])
	ax.set_xticks(np.arange(x_min, x_max, x_major_interval))
	ax.set_xticks(np.arange(x_min, x_max, x_minor_interval), minor=True)
	ax.set_yticks(np.arange(y_min, y_max, y_major_interval))
	ax.grid(which='major', alpha=0.75)
	ax.grid(which='minor', alpha=0.25)
	ax.set_xlabel('Frequency (kHz)')
	ax.set_ylabel('PSD (db/Hz)')
	ax.set_title(title)

# takehome function for 1 and 2
# own Lfilter (convolution) function
def own_lfilter(filter_coeff, sig, zi = None):

	ntaps = len(filter_coeff)
	# sig = concat(zi, sig)

	y = np.zeros(ntaps + len(sig) - 1)
	for n in range(len(y)): # for each output sample
		for k in range(ntaps): # each filter coefficient
			if n-k < len(sig):
				if n-k < 0 and zi is not None:
					y[n] += filter_coeff[k] * zi[n-k]
				elif n-k >= 0:
					y[n] += filter_coeff[k] * sig[n-k]
					# print(str(n) + " " + str(k))

	if zi is not None: return (y[:len(y)-ntaps+1], sig[len(sig)-ntaps+1:])
	else: return y

# takehome function for 1 and 2
# own firwin (impulse response) function
def lpCoeff(Fs, Fc, ntaps):
	normf = Fc/(Fs/2) # Normalise the cutoff
	taps = np.zeros(ntaps)

	for i in range(ntaps):
		if i == (ntaps-1)/2:
			taps[i] = normf
		else:
			taps[i] = normf * ( np.sin(np.pi*normf*(i-(ntaps-1)/2)) ) / ( np.pi*normf*(i-(ntaps-1)/2) )
		taps[i] *= (np.sin(i*np.pi/ntaps))**2

	return taps


# to test against c++ version for gtest, provide 22kHz to 54kHz @240 KSamples/sec
def bpFirwin(Fs, Fb, Fe, num_taps):
	firwin_coeff = signal.firwin(num_taps, [Fb/(Fs/2), Fe/(Fs/2)], window=('hann'), pass_zero="bandpass")

	return firwin_coeff



def logVector(filename: str, data):
	# Define x-axis
	x_axis = np.arange(len(data))

	# Create a structured array with x_axis and data
	structured_data = np.column_stack((x_axis, data))

	# Save the structured array to a .dat file
	np.savetxt(f'../data/{filename}.dat', structured_data, fmt='%d\t%.8e', delimiter='\t', header='x_axis\ty_axis', comments='')

if __name__ == "__main__":
	Fs = 240e3
	Fb = 22e3
	Fe = 54e3
	num_taps = 101

	bp_coeffs = bpFirwin(Fs, Fb, Fe, num_taps)

	logVector("py_firwin_bp", bp_coeffs)