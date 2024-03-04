### General Constraints
- **Input:** 8 bit unsigned, interleaved I and Q samples, ranging from +0 to +255
- **Output:** 16-bit Little Endian signed outputs times 2 channels (Left and Right) with audio data to be fed to `aplay`
	- +RDS output (RDS can be run in modes 0 and 2, it should be ignored in modes 1 and 3)
- Any packet of data must carry samples for no less than **22 ms** and no more than **44 ms**
- Number of partial produces accumulated for the output of a filter must be between **75** and **125**. 

### Group Specific Constraints
 Sample rates of various data points:

| Settings                | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
| ----------------------- | ------ | ------ | ------ | ------ |
| RF Fs (ksamples/sec)    | 2400   | 2304   | 2400   | 2880   |
| IF Fs (ksamples/sec)    | 240    | 384    | 240    | 320    |
| Audio Fs (ksamples/sec) | 48     | 32     | 44.1   | 44.1   |
RDS samples per symbol:

| Settings           | Mode 0 | Mode 2 |
| ------------------ | ------ | ------ |
| Samples per symbol | 13     | 30     |

### Compile Settings
- Enable `-O3` in the `Makefile : CFLAGS`
- If on Pi- use `-mcpu=cortex-a72+crypto`
- If using threads: `Makefile : LDFLAGS = -pthread`
	- If using cmake, add to `CMakeLists.txt : set (PROJECT_LINK_LIBS pthread)`

### [[Runtime]]

