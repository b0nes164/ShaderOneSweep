using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;

public class BlockRadixSortKeyValue : MonoBehaviour
{
    private enum TestType
    {
        //Validates the sort at the current input size on a decreasing sequence of integers
        ValidateSort,

        //Validates the sort at the current input size on a randomly generated bag of integers
        ValidateRandom,

        //Validate that the sort is stable, by initializing keys to all same value, then checking the values
        ValidateStability,

        //Times the execution of a single digit binning pass
        SinglePassTimingTest,

        //Times the execution of the entire algorithm
        AllPassTimingTest,

        //Times the execution of the algorithm using a random input instead of the default sequence of decreasing integers.
        //Because this results in a higher entropy than the default, it tends to have better performance.
        AllPassTimingTestRandom,

        //Executes the algorithm 500 times, then records the execution time in a csv file
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
    private ComputeBuffer sortPayloadBuffer;
    private ComputeBuffer altBuffer;
    private ComputeBuffer altPayloadBuffer;
    private ComputeBuffer globalHistBuffer;
    private ComputeBuffer timingBuffer;

    private const int minSize = 15;
    private const int maxSize = 27;

    private const int k_init = 0;
    private const int k_initRandom = 1;
    private const int k_initStability = 2;
    private const int k_globalHist = 3;
    private const int k_scatterOne = 4;
    private const int k_scatterTwo = 5;
    private const int k_scatterThree = 6;
    private const int k_scatterFour = 7;

    private int radixPasses;
    private int radix;
    private string computeShaderString;

    private uint[] validationArray;

    private int size;
    private bool breaker;

    BlockRadixSortKeyValue()
    {
        radixPasses = 4;
        radix = 256;
        computeShaderString = "BlockRadixSortKeyValue";
    }

    private void Start()
    {
        CheckShader();

        size = 1 << sizeExponent;
        UpdateSize(size);
        UpdateGlobHistBuffer();
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
            case TestType.ValidateStability:
                StartCoroutine(ValidateStability(size));
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
    }

    private void UpdateSortBuffers(int _size)
    {
        if (sortBuffer != null)
            sortBuffer.Dispose();
        if (sortPayloadBuffer != null)
            sortPayloadBuffer.Dispose();
        if (altBuffer != null)
            altBuffer.Dispose();
        if (altPayloadBuffer != null)
            altPayloadBuffer.Dispose();

        sortBuffer = new ComputeBuffer(_size, sizeof(uint));
        sortPayloadBuffer = new ComputeBuffer(_size, sizeof(uint));
        altBuffer = new ComputeBuffer(_size, sizeof(uint));
        altPayloadBuffer = new ComputeBuffer(_size, sizeof(uint));

        compute.SetBuffer(k_init, "b_sort", sortBuffer);
        compute.SetBuffer(k_init, "b_sortPayload", sortPayloadBuffer);

        compute.SetBuffer(k_initRandom, "b_sort", sortBuffer);
        compute.SetBuffer(k_initRandom, "b_sortPayload", sortPayloadBuffer);

        compute.SetBuffer(k_initStability, "b_sort", sortBuffer);
        compute.SetBuffer(k_initStability, "b_sortPayload", sortPayloadBuffer);

        compute.SetBuffer(k_globalHist, "b_sort", sortBuffer);

        compute.SetBuffer(k_scatterOne, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterOne, "b_sortPayload", sortPayloadBuffer);
        compute.SetBuffer(k_scatterOne, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterOne, "b_altPayload", altPayloadBuffer);

        compute.SetBuffer(k_scatterTwo, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterTwo, "b_sortPayload", sortPayloadBuffer);
        compute.SetBuffer(k_scatterTwo, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterTwo, "b_altPayload", altPayloadBuffer);

        compute.SetBuffer(k_scatterThree, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterThree, "b_sortPayload", sortPayloadBuffer);
        compute.SetBuffer(k_scatterThree, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterThree, "b_altPayload", altPayloadBuffer);

        compute.SetBuffer(k_scatterFour, "b_sort", sortBuffer);
        compute.SetBuffer(k_scatterFour, "b_sortPayload", sortPayloadBuffer);
        compute.SetBuffer(k_scatterFour, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterFour, "b_altPayload", altPayloadBuffer);
    }

    private void UpdateGlobHistBuffer()
    {
        globalHistBuffer = new ComputeBuffer(radix * radixPasses, sizeof(uint));

        compute.SetBuffer(k_init, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_initRandom, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_initStability, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_globalHist, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterOne, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterTwo, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterThree, "b_globalHist", globalHistBuffer);
        compute.SetBuffer(k_scatterFour, "b_globalHist", globalHistBuffer);
    }

    private void UpdateTimingBuffer()
    {
        timingBuffer = new ComputeBuffer(1, sizeof(uint));
        compute.SetBuffer(k_scatterOne, "b_timing", timingBuffer);
        compute.SetBuffer(k_scatterFour, "b_timing", timingBuffer);
    }

    private void DispatchKernels()
    {
        compute.Dispatch(k_globalHist, 1, 1, 1);
        compute.Dispatch(k_scatterOne, 1, 1, 1);
        compute.Dispatch(k_scatterTwo, 1, 1, 1);
        compute.Dispatch(k_scatterThree, 1, 1, 1);
        compute.Dispatch(k_scatterFour, 1, 1, 1);
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

    private void ResetBuffersStable()
    {
        compute.Dispatch(k_initStability, 256, 1, 1);
    }

    private IEnumerator ValidateSort(int _size)
    {
        breaker = false;

        validationArray = new uint[_size];
        DispatchKernels();
        sortBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        ValSort(_size, "Keys");
        sortPayloadBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        ValSort(_size, "Values");

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
        ValRand(_size, "Keys");
        sortPayloadBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        ValRand(_size, "Values");

        breaker = true;
    }

    private IEnumerator ValidateStability(int _size)
    {
        breaker = false;

        validationArray = new uint[_size];
        ResetBuffersStable();
        DispatchKernels();
        sortPayloadBuffer.GetData(validationArray);
        yield return new WaitForSeconds(.25f);  //To prevent unity from crashing
        ValSort(_size, "Stability");

        breaker = true;
    }

    private IEnumerator SinglePassTimingTest(int _size)
    {
        breaker = false;

        compute.Dispatch(k_globalHist, 1, 1, 1);
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(globalHistBuffer);
        yield return new WaitUntil(() => request.done);

        float time = Time.realtimeSinceStartup;
        compute.Dispatch(k_scatterOne, 1, 1, 1);
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

        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(globalHistBuffer);
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
        AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(globalHistBuffer);
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

        for (int i = 0; i < 500; ++i)
        {
            compute.SetInt("e_seed", i + 1);
            compute.Dispatch(k_initRandom, 256, 1, 1);
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(globalHistBuffer);
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

    private void ValSort(int _size, string testType)
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
            Debug.Log(testType + " Passed");
        else
            Debug.LogError(testType + " Failed");
    }

    private void ValRand(int _size, string testType)
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
                    Debug.LogError("ERROR AT INDEX " + i + ": " + validationArray[i - 1] + ", " + validationArray[i]);
            }
        }

        if (isValid)
            Debug.Log(testType + " Passed");
        else
            Debug.LogError(testType + " Failed");
    }

    private void OnDestroy()
    {
        if (sortBuffer != null)
            sortBuffer.Dispose();
        if (sortPayloadBuffer != null)
            sortPayloadBuffer.Dispose();
        if (altBuffer != null)
            altBuffer.Dispose();
        if (altPayloadBuffer != null)
            altPayloadBuffer.Dispose();
        if (globalHistBuffer != null)
            globalHistBuffer.Dispose();
        if (timingBuffer != null)
            timingBuffer.Dispose();
    }
}