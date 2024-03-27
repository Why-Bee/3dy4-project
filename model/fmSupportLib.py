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


def fmPll(pllIn, 
          freq, 
          Fs,
          integrator,
          phaseEst,
          feedbackI,
          feedbackQ,
          trigOffset,
          lastNco,
          ncoScale = 1.0, 
          phaseAdjust = 0.0, 
          normBandwidth = 0.01):
    """
    pllIn 	 		array of floats
                    input signal to the PLL (assume known frequency)

    freq 			float
                    reference frequency to which the PLL locks

    Fs  			float
                    sampling rate for the input/output signals

    ncoScale		float
                    frequency scale factor for the NCO output

    phaseAdjust		float
                    phase adjust to be added to the NCO output only

    normBandwidth	float
                    normalized bandwidth for the loop filter
                    (relative to the sampling rate)

    state 			to be added
    """



    # scale factors for proportional/integrator terms
    # these scale factors were derived assuming the following:
    # damping factor of 0.707 (1 over square root of 2)
    # there is no oscillator gain and no phase detector gain

    Cp = 2.666
    Ci = 3.555

    # gain for the proportional term
    Kp = (normBandwidth)*Cp

    # gain for the integrator term
    Ki = (normBandwidth*normBandwidth)*Ci

    # output array for the NCO
    ncoOut = np.empty(len(pllIn)+1)

    # INIITIAL INTERNAL STATE
    # integrator = 0.0
    # phaseEst = 0.0
    # feedbackI = 1.0
    # feedbackQ = 0.0
    # trigOffset = 0
    # lastNco = 1.0

    # NOTE: I have no idea if lastNCO is the right thing to do here.

    ncoOut[0] = lastNco 

    for k in range(len(pllIn)):
        # phase detector
        errorI = pllIn[k] * (+feedbackI)  # complex conjugate of the
        errorQ = pllIn[k] * (-feedbackQ)  # feedback complex exponential

        # four-quadrant arctangent discriminator for phase error detection
        errorD = math.atan2(errorQ, errorI)

        # loop filter
        integrator = integrator + Ki*errorD

        # update phase estimate
        phaseEst = phaseEst + Kp*errorD + integrator

        # internal oscillator

        trigOffset += 1

        trigArg = (2*math.pi*freq/Fs)*(trigOffset) + phaseEst


        feedbackI = math.cos(trigArg)

        feedbackQ = math.sin(trigArg)

        ncoOut[k+1] = math.cos(trigArg*ncoScale + phaseAdjust)

    # for stereo only the in-phase NCO component should be returned

    # for block processing you should also return the state
    # for RDS add also the quadrature NCO component to the output

    return ncoOut, integrator, phaseEst, trigOffset, ncoOut[-1] 

def delayBlock(input_block, state_block):
    output_block = np.concatenate((state_block, input_block[:-len(state_block)]))
    state_block = input_block[-len(state_block):]
    return output_block, state_block

# Clock and data recovery
def sampling_start_adjust(block, samples_per_symbol):
    abs_min_idx = 0
    abs_min = abs(block[abs_min_idx])
    for i in range(0, len(block)-10):
        diff = abs(block[i])
        if diff < abs_min:
            abs_min = diff
            abs_min_idx = i

    return ((abs_min_idx + int(samples_per_symbol/2)) % samples_per_symbol)

def upsample(y, upsampling_factor):
    if upsampling_factor == 1:
        return y

    original_size = len(y)
    y_extended = np.zeros(original_size * upsampling_factor)
    y_extended[::upsampling_factor] = y

    return y_extended

def symbol_vals_to_bits(sampling_points, offset, last_value_state):
    bool_array = np.zeros(int(len(sampling_points)/2), dtype=bool)
    hh_count = 0
    ll_count = 0
    first_val = 0
    second_val = 0
    for i in range(0, len(sampling_points)-1, 2):
        if (i+offset)-1 < 0:
              first_val = last_value_state
        else:
              first_val = sampling_points[(i+offset)-1]
        second_val = sampling_points[(i+offset)]
   
        if first_val == 0 and \
            second_val == 0:
            print("DOUBLE ZERO WARNING")
            bool_array[int(i/2)] = bool(0)

        if first_val >= 0 and \
            second_val <= 0: #case HL
            bool_array[int(i/2)] = bool(1)

        elif first_val <= 0 and \
            second_val >= 0: #case LH
            bool_array[int(i/2)] = bool(0)

        elif first_val < 0 and \
            second_val < 0: #case LL
            if (first_val < second_val): # weak LH
                bool_array[int(i/2)] = bool(0)
            elif (first_val >= second_val): # weak HL
                bool_array[int(i/2)] = bool(1)
            ll_count+=1

        elif first_val > 0 and \
            second_val > 0: #case HH
            if (first_val > second_val): # weak HL
                bool_array[int(i/2)] = bool(1)
            elif (first_val <= second_val): # weak LH
                bool_array[int(i/2)] = bool(0)
            hh_count+=1
            
    return bool_array, ll_count, hh_count

def differential_decode(bool_array,):
    decoded = np.empty(len(bool_array)-1, dtype=bool)
    for i in range(0, len(bool_array)-1):
        decoded[i] = bool_array[i] ^ bool_array[i-1]
    return decoded

def differential_decode_stateful(bool_array, last_val_state):
    decoded = np.empty(len(bool_array)-1, dtype=bool)
    decoded[0] = last_val_state ^ bool_array[0]
    for i in range(1, len(bool_array)-1):
        decoded[i] = bool_array[i] ^ bool_array[i-1]
    return decoded, bool_array[-1]

parity_matrix = np.array(
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
 [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
 [0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
 [1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
 [1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
 [1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
 [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
 [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
 [0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
 [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
 [0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
 [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
 [1, 1, 0, 0, 0, 1, 1, 0, 1, 1]], dtype=bool)

def concat_bool_arr(bool_arr):
    # Convert boolean array to integer
    result = 0
    for bit in bool_arr:
        result = (result << 1) | int(bit)
    return result

def multiply_parity(matrix1):
    if matrix1.ndim != 1:
        raise ValueError("matrix1 must be a 1D array")
    
    result = np.zeros(parity_matrix.shape[1], dtype=bool)
    
    for j in range(parity_matrix.shape[1]):
        for k in range(parity_matrix.shape[0]):
            result[j] ^= matrix1[k] & parity_matrix[k, j]
    
    return concat_bool_arr(result)

def matches_syndrome(ten_bit_val):
    les_syndromes = {
        'A': 0b1111011000,
        'B': 0b1111010100,
        'C': 0b1001011100,
        'Cprime': 0b1111001100,
        'D': 0b1001011000
    }

    for syndrome, value in les_syndromes.items():
        if ten_bit_val == value:
            return True, syndrome
        
    return False, None

def find_frame_start(bitstream, text):
    last_start_idx = 0
    once = 1
    next_up = 0
    missed = 0
    # text = ''
    check_len = 26
    for start_idx in range(0, len(bitstream)-check_len):
        twenty_six_bit_value = bitstream[start_idx:start_idx+check_len]
        ten_bit_code = multiply_parity(twenty_six_bit_value)
        is_valid, syndrome = matches_syndrome(ten_bit_code)
        if is_valid:
            if (start_idx-last_start_idx) < check_len:
                if once:
                    once = 0
                else:
                    missed +=1
                    print("missed one")
                    
            # print(f"{start_idx} {start_idx-last_start_idx} found the syndrome: ", syndrome)
            last_start_idx = start_idx

            if syndrome == 'A':
                print(f"PI: {hex(concat_bool_arr(twenty_six_bit_value[:16]))}")
            if syndrome == 'B':
                print(f"PTY: {(concat_bool_arr(twenty_six_bit_value[6:11]))}")
                next_up = concat_bool_arr(twenty_six_bit_value[:5])
                print(f"Next: {next_up}")
            if syndrome == 'C':
                if next_up == 4:
                    text += chr(concat_bool_arr(twenty_six_bit_value[:8]))
                    text += chr(concat_bool_arr(twenty_six_bit_value[8:16]))
            if syndrome == 'D':
                if next_up == 4:
                    text += chr(concat_bool_arr(twenty_six_bit_value[:8]))
                    text += chr(concat_bool_arr(twenty_six_bit_value[8:16]))
    print("text:", text)
    return text

def recover_bitstream(sampling_points, 
                      bitstream_select, 
                      bitstream_score_0, 
                      bitstream_score_1,
                      last_value_state,
                      bitsteam_select_thresh=20):
    if bitstream_select == 0:
        bitstream, ll_count0, hh_count0 = symbol_vals_to_bits(sampling_points, 0, last_value_state)
    elif bitstream_select == 1:
        bitstream, ll_count1, hh_count1 = symbol_vals_to_bits(sampling_points, 1, last_value_state)
    elif bitstream_select == None:
        bitstream0, ll_count0, hh_count0 = symbol_vals_to_bits(sampling_points, 0, last_value_state)
        bitstream1, ll_count1, hh_count1 = symbol_vals_to_bits(sampling_points, 1, last_value_state)
        # print(f"0: LL count: {ll_count0}, HH count: {hh_count0}")
        # print(f"1: LL count: {ll_count1}, HH count: {hh_count1}")
        if (ll_count0+hh_count0) < (ll_count1+hh_count1):
            bitstream_score_0 += 1
            bitstream_score_1 -= 1
            if bitstream_score_0 >= bitsteam_select_thresh:
                print(f"SELECTING BITSTREAM 0")
                bitstream_select = 0
                bitstream = bitstream0
        else:
            bitstream = bitstream1
            bitstream_score_1 += 1
            bitstream_score_0 -= 1
            if bitstream_score_1 >= bitsteam_select_thresh:
                print(f"SELECTING BITSTREAM 1")
                bitstream_select = 1
    return bitstream, bitstream_select, bitstream_score_0, bitstream_score_1, bitstream[-1]