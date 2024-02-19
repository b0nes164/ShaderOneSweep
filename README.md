# ShaderOneSweep
![Sorting Speeds Comparison(1)](https://github.com/b0nes164/ShaderOneSweep/assets/68340554/7bc88d9d-fce4-48b9-9854-de47ea83b8aa)

# NOTICE: THIS REPO IS OUTDATED AND WILL BE ARCHIVED SOON! IT CONTAINS KNOWN BUGS
# PLEASE USE THE REPO FOUND AT https://github.com/b0nes164/ShaderOneSweep

This project is an HLSL compute shader implementation of the current state of the art GPU sorting algorithm, Adinets and Merrill's [OneSweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), an LSD radix sort that uses Merrill and Garland's [Chained Scan with Decoupled Lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back) to reduce the overall global data movement during a digit-binning pass from $3n$ to $2n$. 

Given an input size of $2^{28}$ 32-bit random keys and a uniform random distribution, our implementation achieves a harmonic mean performance of 5.84 G keys/sec, an effective memory bandwidth utilization of ~44.4%. Although this is lower than the ~55% achieved in the _OneSweep_ paper, the difference is likely due to HLSL's lack of the `reinterpret_cast` functionality that would allow us to easily perform vectorized loading and then transposition to a striped format of the keys. The previous fastest compute shader sorting implementation I could find was by far Dondragmer's [PrefixSort](https://gist.github.com/dondragmer/c75a1a50f1cdd00c104d3483375bdb2f), which is also an 8-bit LSD radix sort and also incorporates Ashkiani et. al.'s [GpuMultiSplit](https://arxiv.org/abs/1701.01189), but uses the older $3n$ data movement Reduce-then-Scan pattern to perform the inter-threadblock prefix sum. To read more about how I performed the testing, see the Testing Methodology section. 

# To Use This Project

1. Download or clone the repository.
2. Drag the contents of `src` into a desired folder within a Unity project.
4. Each sort has a compute shader and a dispatcher. Attach the desired sort's dispatcher to an empty game object. All sort dispatchers are named  `SortNameHere.cs`.
5. Attach the matching compute shader to the game object. All compute shaders are named `SortNameHere.compute`. The dispatcher will return an error if you attach the wrong shader.
6. Ensure the slider is set to a nonzero value.

If you did this correctly you should see this in the inspector:

![image](https://github.com/b0nes164/ShaderOneSweep/assets/68340554/e0e2a00b-60d4-48ed-9101-00fb72a29c10)

<details>

<summary>

## Testing Suite

</summary>

![Suite](https://github.com/b0nes164/ShaderOneSweep/assets/68340554/849e25b5-725e-417f-adfc-3147623e4b75)


Every sort dispatcher has a suite of tests which can be controlled directly from the inspector.

+ `Validate Sort` performs a sort at an input size of $2^{`SizeExponent`}$, with the input being the decreasing series of integers. 

+ `Validate Random` performs a sort at an input size of $2^{`SizeExponent`}$, with the input being a randomly generated set of integers.

+ `Single Pass Timing Test` Times the execution of a single digit binning pass, at an input size of $2^{`SizeExponent`}$, with the input being the decreasing series of integers. 

+ `All Pass Timing Test` Times the execution of the entire sort, at an input size of $2^{`SizeExponent`}$, with the input being the decreasing series of integers. 
  
+ `All Pass Timing Test Random` Times the execution of the entire sort, with the input being a randomly generated set of integers. Because randomly generated integers have a higher entropy than the decreasing sequence of integers, this test demonstrates signficantly better performance.

+ `Record Timing Data` Performs 2000 iterations of `All Pass Timing Test Random`, then logs the results in a `csv` file.

+ `ValidateText` prints any errors during a validation test in the deubg log. This can be quite slow if there are many errors, so it is recommended to also have `QuickText` enabled.

+ `QuickText` limits the number of errors printed during a validation test to 1024.   

</details>


# Important Notes

<details>
  
  <summary>Currently, this project does not work on AMD or integrated graphics hardware.</summary>
  
</br>Unfortunately, AMD, Nvidia, and integrated graphics usually have different wave sizes, which means that code that synchronizes threads on a wave level, like we do, must be manually tuned for each hardware case. Because Unity does not support runtime compilation of compute shaders, we cannot poll the hardware at runtime to compile a targetted shader variant. Although Unity does have the `multi_compile` functionality, it is a very cumbersome solution because it means maintaining and compiling a copy of each kernel for each hardware case.
 
</details>

<details>
  
  <summary>DX12 is a must as well as a minimum Unity version of 2021.1 or later</summary>

</br>As we make heavy use of [WaveIntrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), we need `pragma use_dxc` [to access shader model 6.0](https://forum.unity.com/threads/unity-is-adding-a-new-dxc-hlsl-compiler-backend-option.1086272/).

</details>

# Testing Methodology

To the best of my knowledge, there is no way to time the execution of kernels in-situ in HLSL, so we make do by making an [`AsyncGPUReadbackRequest`](https://docs.unity3d.com/ScriptReference/Rendering.AsyncGPUReadbackRequest.html) on single element buffer, `b_timing` whose sole purpose is to be the target of the readback request. Because `b_timing` is only a single element, any latency introduced by the readback is negligible. So we make a coroutine, make a timestamp, dispatch the kernels, make the readback request, then wait until the readback completes like so: 

```C#
    private IEnumerator AllPassTimingTestRandom(int _size)
    {
        breaker = false;


        ResetBuffersRandom();
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(passHistFour);
        yield return new WaitUntil(() => request.done);


        float time = Time.realtimeSinceStartup;
        compute.Dispatch(k_globalHist, globalHistThreadBlocks, 1, 1);


        compute.Dispatch(k_scatterOne, binningThreadBlocks, 1, 1);


        compute.Dispatch(k_scatterTwo, binningThreadBlocks, 1, 1);


        compute.Dispatch(k_scatterThree, binningThreadBlocks, 1, 1);


        compute.Dispatch(k_scatterFour, binningThreadBlocks, 1, 1);


        request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);
        time = Time.realtimeSinceStartup - time;


        Debug.Log("Raw Time: " + time);
        Debug.Log("Estimated Speed: " + (_size / time) + " ele/sec.");


        breaker = true;
    }
```

To make sure that our initialization kernel is complete, we make an initial readback request beforehand.

## Dondragmer's PrefixSort

Accurately testing Donragmer's [PrefixSort](https://gist.github.com/dondragmer/c75a1a50f1cdd00c104d3483375bdb2f) was challenging because its maximum input size is only $2^{23}$ elements. Because smaller input sizes than $2^{25}$ elements are typically insufficient to fully saturate the memory bandwidth of a GPU, a one-to-one comparison to our _OneSweep_ implementation is misleading, and so we included the $2^{23}$ test in the graph. 

<details>
  
  <summary>To test PrefixSort, we made a modified version of its dispatcher as follows:</summary>

```C#
private IEnumerator RecordTimingData()
    {
        breaker = false;

        Debug.Log("Beginning timing test.");
        List<string> csv = new List<string>();
        ProcessControlsAndEditorSettings();

        int numBlocks = Mathf.CeilToInt(m_numElements / 1024.0f);
        m_sortShader.SetInt("_NumElements", m_numElements);
        m_sortShader.SetInt("_NumBlocks", numBlocks);
        m_sortShader.SetBool("_ShouldSortPayload", m_shouldSortPayload);

        for (int k = 0; k < 500; ++k)
        {
            m_sortShader.SetInt("e_seed", k + 1);
            m_sortShader.SetBuffer(k_init, "KeyOutputBuffer", m_keysBufferA);
            m_sortShader.Dispatch(k_init, 256, 1, 1);
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(m_keysBufferA);
            yield return new WaitUntil(() => request.done);

            float time = Time.realtimeSinceStartup;
            for (int i = 0; i < 4; i++)
            {
                m_sortShader.SetInt("_FirstBitToSort", i * 8);

                //flip the buffers every other sort
                if ((i % 2) == 0)
                {
                    m_sortShader.SetBuffer(m_countTotalsKernel, "KeyInputBuffer", m_keysBufferA);
                    m_sortShader.SetBuffer(m_finalSortKernel, "KeyInputBuffer", m_keysBufferA);
                    m_sortShader.SetBuffer(m_finalSortKernel, "KeyOutputBuffer", m_keysBufferB);

                    m_sortShader.SetBuffer(m_finalSortKernel, "PayloadInputBuffer", m_payloadBufferA);
                    m_sortShader.SetBuffer(m_finalSortKernel, "PayloadOutputBuffer", m_payloadBufferB);
                }
                else
                {
                    m_sortShader.SetBuffer(m_countTotalsKernel, "KeyInputBuffer", m_keysBufferB);
                    m_sortShader.SetBuffer(m_finalSortKernel, "KeyInputBuffer", m_keysBufferB);
                    m_sortShader.SetBuffer(m_finalSortKernel, "KeyOutputBuffer", m_keysBufferA);

                    m_sortShader.SetBuffer(m_finalSortKernel, "PayloadInputBuffer", m_payloadBufferB);
                    m_sortShader.SetBuffer(m_finalSortKernel, "PayloadOutputBuffer", m_payloadBufferA);
                }

                m_sortShader.Dispatch(m_countTotalsKernel, numBlocks, 1, 1);
                m_sortShader.Dispatch(m_blockPostfixKernel, 1, 64, 1);
                m_sortShader.Dispatch(m_calculateOffsetsKernel, numBlocks, 1, 1);
                m_sortShader.Dispatch(m_finalSortKernel, numBlocks, 1, 1);
            }

            request = AsyncGPUReadback.Request(timingBuffer); 
            yield return new WaitUntil(() => request.done);
            time = Time.realtimeSinceStartup - time;

            if (k != 0)
                csv.Add("" + time);

            if ((k & 31) == 0)
                Debug.Log("Running");
        }

        StreamWriter sWriter = new StreamWriter("DonDragmer.csv");
        sWriter.WriteLine("Total Time DonDragmer");
        foreach (string str in csv)
            sWriter.WriteLine(str);
        sWriter.Close();
        Debug.Log("Test Complete");

        breaker = true;
    }
```

</details>

Due to stalling issues with the original method of setting inputs from the CPU side, and to ensure a fair comparison between algorithms, I added an initialization kernel which uses an identical PRNG to the one used in my own implementation. 

<details>
  
  <summary>Because we use identical PRNG's and identical seed values, the inputs for the testing are identical:</summary>

```HLSL
extern int e_seed; //Seed value set from CPU

#define TAUS_STEP_1         ((z1 & 4294967294U) << 12) ^ (((z1 << 13) ^ z1) >> 19)
#define TAUS_STEP_2         ((z2 & 4294967288U) << 4) ^ (((z2 << 2) ^ z2) >> 25)
#define TAUS_STEP_3         ((z3 & 4294967280U) << 17) ^ (((z3 << 3) ^ z3) >> 11)
#define LCG_STEP            (z4 * 1664525 + 1013904223U)
#define HYBRID_TAUS         ((z1 ^ z2 ^ z3 ^ z4) & 268435455)

[numthreads(1024, 1, 1)]
void Init(int3 id : SV_DispatchThreadID)
{
    
    uint z1 = (id.x << 2) * e_seed;
    uint z2 = ((id.x << 2) + 1) * e_seed;
    uint z3 = ((id.x << 2) + 2) * e_seed;
    uint z4 = ((id.x << 2) + 3) * e_seed;
    
    for (int i = id.x; i < _NumElements; i += 1024 * 256)
    {
        z1 = TAUS_STEP_1;
        z2 = TAUS_STEP_2;
        z3 = TAUS_STEP_3;
        z4 = LCG_STEP;
        KeyOutputBuffer[i] = HYBRID_TAUS;
    }
}
```

</details>

# Strongly Suggested Reading and Bibliography
Andy Adinets and Duane Merrill. Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. 2022. arXiv: 2206.01784 [cs.DC]

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Saman Ashkiani et al. “GPU Multisplit”. In: SIGPLAN Not. 51.8 (Feb. 2016). issn: 0362-1340. doi: 10.1145/3016078.2851169. url: https://doi.org/10.1145/3016078.2851169.
