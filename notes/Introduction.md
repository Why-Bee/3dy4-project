### Main Objective
- Create a real-time Software Defined Radio using [[RF dongle]] as input

### FM Tech
- Each FM channel takes up 200 kHz and is centered at any freq from 88.1 MHz to 107.9 MHz.
- In the range \[center :: center+100kHz] we have:
	- [[Mono]]: 0 to 15 kHz
	- [[Stereo]]: 23 to 53 kHz
	- [[RDS]]: 54 to 60 kHz

### Big Picture
![[Pasted image 20240228154652.png]]
- RF hardware acquires the signal. It is output as 8b I sample, 8b Q sample, 8b I sample, 8b Q sample....etc. Run it using [[rtl-sdr]] command. The output sample rate is based on the [[constraints]]
- [[RF Front-end]] gets the FM channel and demodulates it. There are 4 [[modes]] of operation. This block is common for mono, stereo, and RDS path.
- [[Mono]] path
- [[Stereo]] path
- [[RDS]] path

