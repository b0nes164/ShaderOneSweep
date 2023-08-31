using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;

public class OneSweep : MonoBehaviour
{
    private enum TestType
    {
        //Validates the sort at the current input size on a decreasing sequence of integers
        ValidateSort,

        //Validates the sort at the current input size on a randomly generated bag of integers
        ValidateRandom,

        //Times the execution of the a single digit binning pass
        SinglePassTimingTest,

        //Times the execution of the entire algorithm
        AllPassTimingTest,

        //Times the execution of the algorithm using a random input instead of the default sequence of decreasing integers.
        //Because this results in a higher entropy than the default, it tends to have better performance.
        AllPassTimingTestRandom,

        //Executes the algorithm 2000 times, then records the execution time in a csv file
        RecordTimingData,
    }

    [SerializeField]
    private TestType testType;

    [Range(minSize, maxSize)]
    public int sizeExponent;

    [SerializeField]
    private ComputeShader compute;

    [SerializeField]
    private bool validateText;

    [SerializeField]
    private bool quickText;

    private ComputeBuffer sortBuffer;
    private ComputeBuffer altBuffer;
    private ComputeBuffer globalHistBuffer;

    private ComputeBuffer indexBuffer;
    private ComputeBuffer passHistBuffer;
    private ComputeBuffer passHistTwo;
    private ComputeBuffer passHistThree;
    private ComputeBuffer passHistFour;

    private ComputeBuffer timingBuffer;

    private const int minSize = 15;
    private const int maxSize = 28;

    private const int k_init = 0;
    private const int k_initRandom = 1;
    private const int k_globalHist = 2;
    private const int k_scatterOne = 3;
    private const int k_scatterTwo = 4;
    private const int k_scatterThree = 5;
    private const int k_scatterFour = 6;

    private int globalHistThreadBlocks;
    private int radixPasses;
    private int radix;
    private int partitionSize;
    private string computeShaderString;

    private uint[] validationArray;

    private int size;
    private bool breaker;

    OneSweep()
    {
        radixPasses = 4;
        radix = 256;
        globalHistThreadBlocks = 2048;
        partitionSize = 7680;
        computeShaderString = "OneSweep";
    }

    private void Start()
    {
        CheckShader();

        size = 1 << sizeExponent;
        UpdateSize(size);
        UpdateGlobHistBuffer();
        UpdateIndexBuffer();
        UpdateTimingBuffer();
        breaker = true;

        Debug.Log(computeShaderString + ": init Complete.");
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (breaker)
            {
                if (size != (1 << sizeExponent))
                {
                    size = 1 << sizeExponent;
                    UpdateSize(size);
                }

                Dispatcher();
            }
            else
            {
                Debug.LogWarning("Please allow the current test to complete before attempting any other tests.");
            }
        }
    }

    private void Dispatcher()
    {
        ResetBuffers();

        switch (testType)
        {
            case TestType.ValidateSort:
                StartCoroutine(ValidateSort(size));
                break;
            case TestType.ValidateRandom:
                StartCoroutine(ValidateSortRandom(size));
                break;
            case TestType.SinglePassTimingTest:
                StartCoroutine(SinglePassTimingTest(size));
                break;
            case TestType.AllPassTimingTest:
                StartCoroutine(AllPassTimingTest(size));
                break;
            case TestType.AllPassTimingTestRandom:
                StartCoroutine(AllPassTimingTestRandom(size));
                break;
            case TestType.RecordTimingData:
                StartCoroutine(RecordTimingData());
                break;
            default:
                break;
        }
    }

    private void CheckShader()
    {
        try
        {
            compute.FindKernel("Init" + computeShaderString);
        }
        catch
        {
            Debug.LogError("Kernel(s) not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is" + computeShaderString + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }

    private void UpdateSize(int _size)
    {
        compute.SetInt("e_size", _size);
        UpdateSortBuffers(_size);
        UpdatePassHistBuffer(_size);
    }

    private void UpdateSortBuffers(int _size)
    {
        if (sortBuffer != null)
            sortBuffer.Dispose();
        if (altBuffer != null)
            altBuffer.Dispose();

        sortBuffer = new ComputeBuffer(_size, sizeof(uint));
        altBuffer = new ComputeBuffer(_size, sizeof(uint));

        compute.SetBuffer(k_init, "b_sort", sortBuffer);
        compute.SetBuffer(k_initRandom, "b_sort", sortBuffer);
        compute.SetBuffer(k_globalHist, "b_sort", sortBuffer);

        compute.SetBuffer(k_scatterOne, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterOne, "b_alt", altBuffer);

        compute.SetBuffer(k_scatterTwo, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterTwo, "b_alt", altBuffer);

        compute.SetBuffer(k_scatterThree, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterThree, "b_alt", altBuffer);

        compute.SetBuffer(k_scatterFour, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterFour, "b_alt", altBuffer);
    }

    private void UpdatePassHistBuffer(int _size)
    {
        if (passHistBuffer != null)
            passHistBuffer.Dispose();
        if (passHistTwo != null)
            passHistTwo.Dispose();
        if (passHistThree != null)
            passHistThree.Dispose();
        if (passHistFour != null)
            passHistFour.Dispose();

        passHistBuffer = new ComputeBuffer(_size / partitionSize * radix, sizeof(uint));
        passHistTwo = new ComputeBuffer(_size / partitionSize * radix, sizeof(uint));
        passHistThree = new ComputeBuffer(_size / partitionSize * radix, sizeof(uint));
        passHistFour = new ComputeBuffer(_size / partitionSize * radix, sizeof(uint));

        //init
        compute.SetBuffer(k_init, "b_passHist", passHistBuffer);
        compute.SetBuffer(k_init, "b_passTwo", passHistTwo);
        compute.SetBuffer(k_init, "b_passThree", passHistThree);
        compute.SetBuffer(k_init, "b_passFour", passHistFour);

        //init random
        compute.SetBuffer(k_initRandom, "b_passHist", passHistBuffer);
        compute.SetBuffer(k_initRandom, "b_passTwo", passHistTwo);
        compute.SetBuffer(k_initRandom, "b_passThree", passHistThree);
        compute.SetBuffer(k_initRandom, "b_passFour", passHistFour);

        //scatters
        compute.SetBuffer(k_scatterOne, "b_passHist", passHistBuffer);
        compute.SetBuffer(k_scatterTwo, "b_passTwo", passHistTwo);
        compute.SetBuffer(k_scatterThree, "b_passThree", passHistThree);
        compute.SetBuffer(k_scatterFour, "b_passFour", passHistFour);
    }

    private void UpdateGlobHistBuffer()
    {
        globalHistBuffer = new ComputeBuffer(radix * radixPasses, sizeof(uint));

        compute.SetBuffer(k_init, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_initRandom, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_globalHist, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterOne, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterTwo, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterThree, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterFour, "b_globalHist", globalHistBuffer);
    }
    private void UpdateIndexBuffer()
    {
        indexBuffer = new ComputeBuffer(radixPasses, sizeof(uint));

        compute.SetBuffer(k_init, "b_index", indexBuffer);
        compute.SetBuffer(k_initRandom, "b_index", indexBuffer);

        compute.SetBuffer(k_scatterOne, "b_index", indexBuffer);
        compute.SetBuffer(k_scatterTwo, "b_index", indexBuffer);
        compute.SetBuffer(k_scatterThree, "b_index", indexBuffer);
        compute.SetBuffer(k_scatterFour, "b_index", indexBuffer);
    }
    private void UpdateTimingBuffer()
    {
        timingBuffer = new ComputeBuffer(1, sizeof(uint));
        compute.SetBuffer(k_scatterOne, "b_timing", timingBuffer);
        compute.SetBuffer(k_scatterFour, "b_timing", timingBuffer);
    }

    private void DispatchKernels()
    {
        compute.Dispatch(k_globalHist, globalHistThreadBlocks, 1, 1);
        compute.Dispatch(k_scatterOne, size / partitionSize, 1, 1);
        compute.Dispatch(k_scatterTwo, size / partitionSize, 1, 1);
        compute.Dispatch(k_scatterThree, size / partitionSize, 1, 1);
        compute.Dispatch(k_scatterFour, size / partitionSize, 1, 1);
    }

    private void ResetBuffers()
    {
        compute.Dispatch(k_init, 256, 1, 1);
    }

    private void ResetBuffersRandom()
    {
        compute.SetInt("e_seed", (int)(Time.realtimeSinceStartup * 100000.0f));
        compute.Dispatch(k_initRandom, 256, 1, 1);
    }

    private IEnumerator ValidateSort(int _size)
    {
        breaker = false;

        validationArray = new uint[_size];
        DispatchKernels();
        sortBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        ValSort(_size);

        breaker = true;
    }

    private IEnumerator ValidateSortRandom(int _size)
    {
        breaker = false;

        validationArray = new uint[_size];
        ResetBuffersRandom();
        DispatchKernels();
        sortBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        ValRand(_size);

        breaker = true;
    }

    private IEnumerator SinglePassTimingTest(int _size)
    {
        breaker = false;

        compute.Dispatch(k_globalHist, globalHistThreadBlocks, 1, 1);
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(globalHistBuffer);
        yield return new WaitUntil(() => request.done);

        float time = Time.realtimeSinceStartup;
        compute.Dispatch(k_scatterOne, _size / partitionSize, 1, 1);
        request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);
        time = Time.realtimeSinceStartup - time;

        Debug.Log("Raw Time: " + time);
        Debug.Log("Estimated Speed: " + (_size / time) + " ele/sec.");

        breaker = true;
    }

    private IEnumerator AllPassTimingTest(int _size)
    {
        breaker = false;

        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(passHistFour);
        yield return new WaitUntil(() => request.done);

        float time = Time.realtimeSinceStartup;
        DispatchKernels();
        request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);
        time = Time.realtimeSinceStartup - time;

        Debug.Log("Raw Time: " + time);
        Debug.Log("Estimated Speed: " + (_size / time) + " ele/sec.");

        breaker = true;
    }

    private IEnumerator AllPassTimingTestRandom(int _size)
    {
        breaker = false;

        ResetBuffersRandom();
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(passHistFour);
        yield return new WaitUntil(() => request.done);

        float time = Time.realtimeSinceStartup;
        DispatchKernels();
        request = AsyncGPUReadback.Request(timingBuffer);
        yield return new WaitUntil(() => request.done);
        time = Time.realtimeSinceStartup - time;

        Debug.Log("Raw Time: " + time);
        Debug.Log("Estimated Speed: " + (_size / time) + " ele/sec.");

        breaker = true;
    }

    public IEnumerator RecordTimingData()
    {
        breaker = false;
        Debug.Log("Beginning timing test, this may take a while.");


        List<string> csv = new List<string>();
        float time;

        for (int i = 0; i < 2000; ++i)
        {
            compute.SetInt("e_seed", i + 1);
            compute.Dispatch(k_initRandom, 256, 1, 1);
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(passHistFour);
            yield return new WaitUntil(() => request.done);

            time = Time.realtimeSinceStartup;
            DispatchKernels();
            request = AsyncGPUReadback.Request(timingBuffer);
            yield return new WaitUntil(() => request.done);
            time = Time.realtimeSinceStartup - time;

            if (i != 0)
                csv.Add("" + time);

            if ((i & 31) == 0)
                Debug.Log("Running");
        }

        StreamWriter sWriter = new StreamWriter(computeShaderString + ".csv");
        sWriter.WriteLine("Total Time " + computeShaderString + ":");
        foreach (string str in csv)
            sWriter.WriteLine(str);
        sWriter.Close();
        Debug.Log("Test Complete");

        breaker = true;
    }

    private void ValSort(int _size)
    {
        bool isValid = true;
        int errCount = 0;

        for (uint i = 0; i < _size; ++i)
        {
            if (validationArray[i] != i + 1)
            {

                if (isValid)
                    isValid = false;

                if (quickText)
                    errCount++;

                if (errCount < 1024)
                    Debug.LogError("EXPECTED SAME AT INDEX " + i + ": " + (i + 1) + ", " + validationArray[i]);
            }
        }

        if (isValid)
            Debug.Log("Sort Passed");
        else
            Debug.LogError("Sort Failed");
    }

    private void ValRand(int _size)
    {
        bool isValid = true;
        int errCount = 0;

        for (int i = 1; i < _size; ++i)
        {
            if (validationArray[i] < validationArray[i - 1])
            {

                if (isValid)
                    isValid = false;

                if (quickText)
                    errCount++;

                if (errCount < 1024)
                    Debug.LogError("EXPECTED SAME AT INDEX " + i + ": " + (i + 1) + ", " + validationArray[i]);
            }
        }

        if (isValid)
            Debug.Log("Sort Passed");
        else
            Debug.LogError("Sort Failed");
    }

    private void OnDestroy()
    {
        if (sortBuffer != null)
            sortBuffer.Dispose();
        if (altBuffer != null)
            altBuffer.Dispose();
        if (globalHistBuffer != null)
            globalHistBuffer.Dispose();
        if (indexBuffer != null)
            indexBuffer.Dispose();

        if (passHistBuffer != null)
            passHistBuffer.Dispose();
        if (passHistTwo != null)
            passHistTwo.Dispose();
        if (passHistThree != null)
            passHistThree.Dispose();
        if (passHistFour != null)
            passHistFour.Dispose();

        if (timingBuffer != null)
            timingBuffer.Dispose();
    }
}