![[Pasted image 20240228163017.png]]
- If there is no integer scale factor that can decimate eg: if it has to be 2.5 M to 48 k, we need to first expand, then decimate.
- Expander logic: If expanding factor is U, put (U-1) zeros after each data point.
- Remember that in the above low pass filter due to the increase in Sample Rate caused by zero padding, both the impulse response size (number of filter taps) and gain must be scaled up by factor U. Just like we can optimize the decimator and LPF to work together, we can also use the fact that there are zeros when working with the low pass filter for optimization
- The Mono data is combined with [[Stereo]] to produce output. Therefore, we need a delay since Stereo is slower [[Stereo#All-Pass Filter on the Mono Path]]