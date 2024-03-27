#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

################### IMPORTS ####################################

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import fmDemodArctan, fmPlotPSD, own_lfilter, lpCoeff, custom_fm_demod, \
	  logVector, bpFirwin, fmPll, delayBlock, sampling_start_adjust, upsample, symbol_vals_to_bits, \
	  differential_decode, multiply_parity, matches_syndrome, find_frame_start
from fmRRC import impulseResponseRootRaisedCosine

################### END IMPORTS ####################################
################### SETTINGS ####################################

mode = 0

iq_factor = 2

rf_Fs = 2.4e6
rf_Fc = 100e3
rf_taps = 101
rf_decim = 10

audio_taps = 101
audio_Fc = 16e3
audio_decim = 5
audio_Fs = 48e3

rds_bpf_fc_low = 54e3
rds_bpf_fc_high = 60e3
rds_bpf_taps = 101

rds_lpf_fc = 3e3
rds_lpf_taps = 101

rds_rrc_taps = 151

rds_squared_bpf_fc_low = 113.5e3
rds_squared_bpf_fc_high = 114.5e3
rds_carrier_frequency = 114e3
rds_squared_bpf_taps = 101

symbols_per_block = None

if mode == 0:
	samples_per_symbol = 13
	symbols_per_block = 38
	rds_downsampling_factor = 1920
	rds_upsampling_factor = 247
	block_size = 2*iq_factor*rf_decim*rds_downsampling_factor
elif mode == 2:
	samples_per_symbol = 30
	rds_upsampling_factor = 19
	rds_downsampling_factor = 64
	audio_downsample_factor = 800
	block_size = 2*3*iq_factor*rf_decim*audio_downsample_factor
else:
	print("INVALID MODE", mode)
	exit(1)
    
rds_decim = rds_downsampling_factor/rds_upsampling_factor

sampling_start_offset = 0

num_blocks_for_pll_tuning = 20
num_blocks_for_cdr = 10

bitstream_select = None
bitstream_score_0 = 0
bitstream_score_1 = 0
bitsteam_select_thresh = 1
text = ''


################### END SETTINGS ####################################
################### STATE VARIABLES ####################################

state_i_lpf_100k = np.zeros(rf_taps-1)
state_q_lpf_100k = np.zeros(rf_taps-1)
rds_filt_state_bpf = np.zeros(rds_bpf_taps-1)
rds_filt_squared_state_bpf = np.zeros(rds_squared_bpf_taps-1)
rds_filt_state_apf = np.zeros(int((rds_bpf_taps-1)/2))
rds_filt_upsample_state_lpf = np.zeros(rds_lpf_taps-1)
rds_filt_mixed_state_lpf = np.zeros(rds_lpf_taps-1)
rds_filt_state_rrc = np.zeros(rds_rrc_taps-1)
ausio_filt_state_lpf = np.zeros(audio_taps-1)

state_phase = np.float64(0.0)
state_i_custom = np.float64(0.0)
state_q_custom = np.float64(0.0)

# INIITIAL PLL STATES
pll_state_integrator = 0.0
pll_state_phaseEst = 0.0
pll_state_feedbackI = 1.0
pll_state_feedbackQ = 0.0
pll_state_trigOffset = 0
pll_state_lastNco = 1.0

pll_state_integrator_q =  0.0
pll_state_phaseEst_q =  0.0
pll_state_feedbackI_q =  1.0
pll_state_feedbackQ_q =  0.0
pll_state_trigOffset_q =  0
pll_state_lastNco_q =  1.0

audio_data = np.array([])
sampling_points_aggregated = np.array([])

################### END STATE VARIABLES ####################################

if __name__ == "__main__":

################### READ IN DATA ####################################
	in_fname = "../data/samples9.raw"

	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0)/128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

################### END READ IN DATA ####################################
################### CALCULATE COEFFS ####################################

	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

	audio_coeff = signal.firwin(audio_taps, audio_Fc/((rf_Fs/rf_decim)/2), window=('hann'))

	rds_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
								rds_bpf_fc_low, 
								rds_bpf_fc_high, 
								rds_bpf_taps)

	rds_squared_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
										rds_squared_bpf_fc_low, 
										rds_squared_bpf_fc_high, 
										rds_squared_bpf_taps)

	rds_lpf_coeff = signal.firwin(rds_lpf_taps, rds_lpf_fc/((rf_Fs/rf_decim)/2), window=('hann'))



	rds_rrc_coeff = impulseResponseRootRaisedCosine(rf_Fs/(rf_decim*rds_decim), rds_rrc_taps)

################### END CALCULATE COEFFS ####################################
################### GET BLOCK ####################################
	
	num_iter = len(iq_data) // block_size
	
	for block_count in range(0, num_iter):
		# print('Processing block ' + str(block_count))

################### END GET BLOCK ####################################
################### RF FRONT ####################################

		# filter to extract the FM channel (I samples are even, Q samples are odd)
		i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
				zi=state_i_lpf_100k)

		q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
				zi=state_q_lpf_100k)

		# downsample the I/Q data from the FM channel
		i_ds = i_filt[::rf_decim]
		q_ds = q_filt[::rf_decim]

		# FM demodulator
		fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)

################### END RF FRONT ####################################
################### AUDIO CHECK ####################################
		
		# audio_filt, ausio_filt_state_lpf = signal.lfilter(audio_coeff, \
		# 											 1.0, fm_demod, zi = ausio_filt_state_lpf)
		# audio_data = np.concatenate((audio_data, audio_filt[::audio_decim]))
		# out_fname = "../data/fmRdsBlock_mono.wav"

		# if (block_count == num_iter-1):
		# 	wavfile.write(out_fname, int(audio_Fs), np.int16((audio_data/2)*32767))

################### END AUDIO CHECK ####################################
################### RDS WAVEFORM EXTRACTION ####################################

		# Bandpass filter
		rds_filt, rds_filt_state_bpf = signal.lfilter(rds_bpf_coeff,\
												1.0, fm_demod, zi=rds_filt_state_bpf)

		# square signal
		rds_filt_squared = rds_filt*rds_filt

		rds_filt_carrier, rds_filt_squared_state_bpf = signal.lfilter(rds_squared_bpf_coeff,\
																1.0, rds_filt_squared, zi=rds_filt_squared_state_bpf)

		ncoOut_inPhase, pll_state_integrator, pll_state_phaseEst, pll_state_trigOffset, pll_state_lastNco = fmPll(
					rds_filt_carrier, 
					rds_carrier_frequency, 
					rf_Fs/rf_decim, 
					pll_state_integrator, 
					pll_state_phaseEst,
					pll_state_feedbackI,
					pll_state_feedbackQ,
					pll_state_trigOffset, 
					pll_state_lastNco, 
					ncoScale=0.5,
					normBandwidth=0.0025)
		
		if (block_count < num_blocks_for_pll_tuning):
			continue
		# elif ( block_count == num_blocks_for_pll_tuning):
		# 	plt.plot(ncoOut_inPhase[0:100])
		# 	plt.plot(100*rds_filt_carrier[0:100])
		# 	plt.show()

################### END RDS WAVEFORM EXTRACTION ####################################
################### ALL PASS FILTER RDS DATA ####################################
		
		rds_filt_delayed, rds_filt_state_apf = delayBlock(rds_filt, rds_filt_state_apf)

################### END ALL PASS FILTER RDS DATA ####################################
################### RDS DATA DEMODULATION ####################################
		
		# Mixer!! credit @Ivan Lange, langei@mcmaster.ca
		rds_mixed = 2 * rds_filt_delayed * ncoOut_inPhase[:-1]

		rds_mixed_lfiltered, rds_filt_mixed_state_lpf = signal.lfilter(rds_lpf_coeff, \
									    1.0, rds_mixed, zi=rds_filt_mixed_state_lpf )

		rds_upsampled = upsample(rds_mixed_lfiltered, rds_upsampling_factor)

		rds_upsampled_lfilterd, rds_filt_upsample_state_lpf = signal.lfilter(rds_lpf_coeff, \
										   1.0, rds_upsampled, zi=rds_filt_upsample_state_lpf )

		rds_resampled = rds_upsampling_factor * rds_upsampled_lfilterd[::rds_downsampling_factor]

		rds_rrcfiltered, rds_filt_state_rrc = signal.lfilter(rds_rrc_coeff, \
								1.0, rds_resampled, zi=rds_filt_state_rrc)

################### END RDS DATA DEMODULATION ####################################
################### CLOCK AND DATA RECOVERY ####################################
		if (block_count < num_blocks_for_pll_tuning+num_blocks_for_cdr):
			sampling_start_offset += sampling_start_adjust(rds_rrcfiltered, samples_per_symbol)
			continue
		elif (block_count == num_blocks_for_pll_tuning+num_blocks_for_cdr):
			sampling_start_offset = sampling_start_offset//num_blocks_for_cdr
			print("sampling start offset: ", sampling_start_offset)
		# if (block_count == 200):
		# 	sampling_points_graphing = upsample(np.ones(symbols_per_block), samples_per_symbol)[:-(sampling_start_offset)]
		# 	sampling_points_graphing = np.concatenate((np.zeros(sampling_start_offset), sampling_points_graphing))
		# 	sampling_points_graphing = sampling_points_graphing * rds_rrcfiltered

		# 	plt.plot(sampling_points_graphing)
		# 	plt.plot(rds_rrcfiltered)
		# 	plt.show() 

		sampling_points = rds_rrcfiltered[sampling_start_offset::samples_per_symbol]

		sampling_points_aggregated = np.concatenate((sampling_points_aggregated, sampling_points))
		if block_count % 311 != 0:
			if block_count % 311 == 1 and block_count != 1:
				sampling_points_aggregated = sampling_points
			continue

################### END CLOCK AND DATA RECOVERY ####################################
################### RECOVER BITSTREAM ####################################
		if bitstream_select == None:
			bitstream0, ll_count0, hh_count0 = symbol_vals_to_bits(sampling_points_aggregated, 0)
			bitstream1, ll_count1, hh_count1 = symbol_vals_to_bits(sampling_points_aggregated, 1)

			print(f"0: LL count: {ll_count0}, HH count: {hh_count0}")
			print(f"1: LL count: {ll_count1}, HH count: {hh_count1}")

			# pick the right bitstream
			if (ll_count0+hh_count0) < (ll_count1+hh_count1):
				bitstream_score_0 += 1
				bitstream_score_1 -= 1
				if bitstream_score_0 >= bitsteam_select_thresh:
					print(f"SELECTING BITSTREAM 0: score0={bitstream_score_0}, score1={bitstream_score_1}")
					bitstream_select = 0
					bitstream  = bitstream0
			else:
				bitstream  = bitstream1
				bitstream_score_1 += 1
				bitstream_score_0 -= 1
				if bitstream_score_1 >= bitsteam_select_thresh:
					print(f"SELECTING BITSTREAM 1: score0={bitstream_score_0}, score1={bitstream_score_1}")
					bitstream_select = 1
		elif bitstream_select == 1:
			bitstream, ll_count1, hh_count1 = symbol_vals_to_bits(sampling_points_aggregated, 1)
		elif bitstream_select == 0:
			bitstream, ll_count0, hh_count0 = symbol_vals_to_bits(sampling_points_aggregated, 0)

################### END RECOVER BITSTREAM ####################################
################### DIFFERENTIAL DECODING ####################################

		bitstream_decoded = differential_decode(bitstream)

################### END DIFFERENTIAL DECODING ####################################
################### FRAME SYNCHRONIZATION ####################################
		text = find_frame_start(bitstream_decoded, text)

# if __name__ == "__main__":

# 	# read the raw IQ data from the recorded file
# 	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
# 	in_fname = "../data/samples9.raw"

# 	raw_data = np.fromfile(in_fname, dtype='uint8')
# 	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
# 	# IQ data is normalized between -1 and +1 in 32-bit float format
# 	iq_data = (np.float32(raw_data) - 128.0)/128.0
# 	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")
	
# 	# set up the subfigures for plotting
# 	subfig_height = np.array([0.8, 2, 1.6])
# 	plt.rc('figure', figsize=(7.5, 7.5))	
# 	fig, (ax0) = plt.subplots(nrows=1, gridspec_kw={'height_ratios': subfig_height})
# 	fig.subplots_adjust(hspace = .6)

# 	# coefficients for the front-end low-pass filter
# 	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))
	
# 	rds_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
#                               rds_bp_fc_low, 
#                               rds_bp_fc_high, 
#                               rds_bp_taps)
	
# 	rds_squared_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
#                                       rds_squared_bp_fc_low, 
#                                       rds_squared_bp_fc_high, 
#                                       rds_squared_bp_taps)
	
# 	# State variables
# 	state_i_lpf_100k = np.zeros(rf_taps-1)
# 	state_q_lpf_100k = np.zeros(rf_taps-1)

# 	# select a block_size that is a multiple of KB
# 	# and a multiple of decimation factors
# 	block_size = 1024 * rf_decim * 10 * 2
# 	block_count = 0

# 	# if the number of samples in the last block is less than the block size
# 	# it is fine to ignore the last few samples from the raw IQ file
# 	while (block_count+1)*block_size < len(iq_data):

# 		# if you wish to have shorter runtimes while troubleshooting
# 		# you can control the above loop exit condition as you see fit
# 		print('Processing block ' + str(block_count))

# 		# filter to extract the FM channel (I samples are even, Q samples are odd)
# 		i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
# 				iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
# 				zi=state_i_lpf_100k)

# 		q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
# 				iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
# 				zi=state_q_lpf_100k)

# 		# downsample the I/Q data from the FM channel
# 		i_ds = i_filt[::rf_decim]
# 		q_ds = q_filt[::rf_decim]

# 		# FM demodulator
# 		fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)
		
#         # filter rds data
# 		rds_filt, rds_filt_state = signal.lfilter(rds_bpf_coeff, 1.0, fm_demod, zi=rds_filt_state)
		
#         # squaring non-linearity
# 		rds_filt_squared = rds_filt*rds_filt
		
#         # extract the 114 kHz carrier
# 		rds_filt_carrier, rds_filt_carrier_state = signal.lfilter(rds_squared_bpf_coeff, 1.0, rds_filt_squared, zi=rds_filt_carrier_state)
		
# 		ncoOut, pll_state_integrator, pll_state_phaseEst, pll_state_trigOffset, pll_state_lastNco = fmPll(
# 			rds_filt_carrier, 
# 			114e3, 
# 			rf_Fs/rf_decim, 
# 			pll_state_integrator, 
# 			pll_state_phaseEst,
# 			pll_state_feedbackI,
# 			pll_state_feedbackQ,
# 			pll_state_trigOffset, 
# 			pll_state_lastNco, 
# 			ncoScale=0.5,
# 			normBandwidth=0.0025)
		
# 		if block_count == 20:
# 			plt.plot(ncoOut[:200])
# 			plt.plot(rds_filt_carrier[:200]*75)
# 			plt.show()
			
# 		block_count += 1

# 	# loop end
# 	print('Finished processing all the blocks from the recorded I/Q samples')

