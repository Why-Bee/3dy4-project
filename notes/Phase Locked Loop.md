- Software PLL can be used for most frequency ranges, we don't need a hardware PLL.
- Main goal of a PLL is to produce a clean oscillatory wave, even if the input wave is noisy. 
- This clean wave is of equal frequency and in-phase with the input wave. this is called **locked wave**
- The faster a PLL can lock, the more phase jitter exists in the output, and the worse the ability of the PLL to lock on a noisy output.
![[Pasted image 20240304162002.png]]
- Code for PLL and NCO is given in Python
