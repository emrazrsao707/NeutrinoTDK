I used 3flav_calc to calculate and visualize neutrino flavor-transition probabilities in three-flavor scenarios. 2flav_LE is used to visualize two-flavor neutrino flavor-transition probabilities.

The program uses Python’s NumPy library and follows its number format. When entering inputs, use a period (.) rather than a comma (,) as the decimal separator. For scientific notation, use a format such as 0.78e-4, where e indicates the exponent. In NumPy, multiplication is written with *, and exponentiation with **.

Several density profiles can be used to model oscillations in matter. These can be changed in the density_profile section of the code. Currently, only constant, linear, and exponential profiles are supported. In 3flav_calc, you can change the L/E values for which the flavor-transition probabilities are calculated in the variables list.

This is the exact version of the code I used for my TDK (Scientific Students’ Associations Conference; OTDK) paper, submitted in November 2026. I plan to continue developing the code in the future.
