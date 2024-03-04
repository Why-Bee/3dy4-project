![[Pasted image 20240304155845.png]]
- Stereo channel (after filtering) carries the difference between left and right audio channels.
- The signal is on a 38 kHz carrier. It is an AM signal using double sideband suppressed carrier.
	- Bandwidth W = 15 kHz
	- Center Frequency = 38 kHz
![[Pasted image 20240304160009.png]]
- Input: the demod signal at the IF sample rate.
- Output: audio sample that can be combined with mono to produce the left and right audio channels.
#### Stereo Channel Extraction and Carrier Recovery
- Recall 3TR4: DSB-SC multiplies message signal with carrier signal cos(2$\pi f_c$)
- First we need to do **Stereo Channel Extraction**
	- It takes the stereo part between 22 kHz and 54 kHz
- To do the demodulation, we need to mix the stereo channel with the **stereo carrier**
	- This is done by extracting the 19 kHz pilot tone, whose second harmonic is at 38 kHz. This is fed into [[Phase Locked Loop]] which outputs synchronised 19 kHz tone, then we use a numerically controlled oscillator to multiply the frequency by 2 to produce the stereo carrier
- Then we take the stereo carrier, mix it with the stereo channel, and it is mixed as shown in above 
- The output of the **Mixer** in the receiver produces: $\frac{A_m}{2} cos(2\pi f_m ) + \frac{A_m}{4} [cos(2 \pi (2f_c + f_m )) + cos(2\pi (2f_c - f_m))]$
	- Here, first term is original message, and the second term has a high frequency cosines. Therefore, if we low pass filter the output from the mixer we will get the old message back.
- We will need to make a function that can produce band-pass filter taps (currently we can only produce low-pass filter)
![[Pasted image 20240304161938.png]]
#### Mixing and Stereo Processing
![[Pasted image 20240304162307.png]]
- The mixing pointwise multiplies the samples from the recovered carrier, and the recovered signal.
- Then the sample rate conversion is a low pass filter and decimator.

#### All-Pass Filter on the Mono Path
- We also need to consider that the [[Mono]] path is usually processed much faster than the stereo path. So we will have an error on the stereo combiner
	- to fix this, we can add an all-pass filter on mono path to match the delay for stereo extraction.
- **Math:** For a filter, the phase delay of a filter is $\frac{1}{Fs} \cdot \frac {N-1}{2}$ where Fs = sample rate, N = taps.
- Therefore, to match delays when the signal branches and reconverges, the band pass filters in [[Stereo#Stereo Channel Extraction and Carrier Recovery]] should have the same number of filter taps.
- We also need to use this property to add all pass filter with delay equal to the delay of these stereo filters, n the mono channnel.
- This is not needed for RDS as the RDS path never converges with the audio paths.
