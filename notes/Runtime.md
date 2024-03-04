 Code must accept input on Cin, and output to Cout
- If samples already exist:
	`cat iq_samples.raw | ./project {mode} {m/s/r} | aplay -f S16_LE -c {1/2} -r {Audio_Fs}`
- If samples must be generated:
	`rtl_sdr -f {channel_freq} -s {RF_Fs} -n number_of_samples} - > iq_samples.raw`
- If running real-time:
	`rtl_sdr -f {channel_freq} -s {RF_Fs - | ./project {mode} {m/s/r} | aplay -f S16_LE -c {1/2} -r {Audio_Fs}`