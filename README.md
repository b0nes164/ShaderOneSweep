# ShaderOneSweep
![Sorting Speeds Comparison](https://github.com/b0nes164/ShaderOneSweep/assets/68340554/33387b57-387c-4aab-aac0-fc674b67c0e6)


This project is an HLSL compute shader implementation of the current state of the art GPU sorting algorithm, Adinets and Merrill's [OneSweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), an LSD radix sort that uses Merrill and Garland's [Chained Scan with Decoupled Lookback] to reduce the overall global data movement during a digit-binning pass from $3n$ to $2n$. Given an input size of $2^{28}$ 32-bit random keys and a uniform random distribution, our implementation achieves a harmonic mean performance of 5.84 G keys/sec, an effective memory bandwidth utilization of ~42.4%. Although this is lower than the ~55% achieved in the _OneSweep_ paper, the difference is likely due to HLSL's lack of the `reinterpret_cast` functionality that would allow us to easily perform vectorized loading and then transposition to a striped format of the keys.

# To Use This Project

Instructions here.

# Important Notes

Notes here.

# Testing Methodology

testing methodology here.
