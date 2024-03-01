![[Pasted image 20240228162424.png]]
- **Any filter followed by a decimator should compute only the samples that are kept after decimation**
- No matter the mode, this code will not change any values since the decimator decimation rate is identical and the 
- I-Q samples coming in will be 8 bit unsigned ints. Values need to be normalized to range \[-1, 1]
- Lab code assumes float data type, but we can use double which may or may not be needed.
- [[constraints]] are the 8 bit unsigned integer value inputs, and the 16 bit signed integer values outputs. Internal data representation and normalizing is up to our group.
