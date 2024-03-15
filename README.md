
# NOTICE: This repository has been archived.
This repository has been archived. The development and maintenance of its contents have been moved to [https://github.com/b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting).

# ShaderOneSweep
This project is an HLSL compute shader implementation of the current state-of-the-art GPU sorting algorithm, Adinets and Merrill's [OneSweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), an LSD radix sort that uses Merrill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back) to reduce the overall global data movement during a digit-binning pass from $3n$ to $2n$.

Given an input size of $2^{28}$ 32-bit uniform random keys and a 2080 Super, this implementation achieves a harmonic mean performance of 9.55 G keys/sec as opposed to the 10.9 G keys/sec achieved in the CUDA [CUB](https://github.com/NVIDIA/cccl) library.

# To Use This Project
1. Download or clone the repository.
2. Drag the contents of `src` into a desired folder within a Unity project.
3. Each sort has a compute shader and a dispatcher. Attach the desired sort's dispatcher to an empty game object. All sort dispatchers are named `SortNameHere.cs`.
4. Attach the matching compute shader to the game object. All compute shaders are named `SortNameHere.compute`. The dispatcher will return an error if you attach the wrong shader.
5. Ensure the slider is set to a non-zero value.

# Strongly Suggested Reading and Bibliography
- Andy Adinets and Duane Merrill. Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. 2022. arXiv: 2206.01784. url: [https://arxiv.org/abs/2206.01784](https://arxiv.org/abs/2206.01784)
- Duane Merrill and Michael Garland. "Single-pass Parallel Prefix Scan with De-coupled Lookback". 2016. url: [https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
- Saman Ashkiani et al. "GPU Multisplit". In: SIGPLAN Not. 51.8 (Feb. 2016). issn: 0362-1340. doi: [10.1145/3016078.2851169](https://doi.org/10.1145/3016078.2851169). url: [https://doi.org/10.1145/3016078.2851169](https://doi.org/10.1145/3016078.2851169).