using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OneSweepKeyValueHelper : MonoBehaviour
{
    [SerializeField]
    private ComputeShader compute;


    private const int radixPasses = 4;
    private const int radix = 256;
    private const int globalHistThreadBlocks = 2048;
    private const int partitionSize = 7680;
    private const string computeShaderString = "OneSweepKeyValue";

    private const int k_init = 0;
    private const int k_globalHist = 4;
    private const int k_scatterOne = 5;
    private const int k_scatterTwo = 6;
    private const int k_scatterThree = 7;
    private const int k_scatterFour = 8;

    private int buffer_size;
    private ComputeBuffer altBuffer, payloadAltBuffer;

    private ComputeBuffer passHistBuffer;
    private ComputeBuffer passHistTwo;
    private ComputeBuffer passHistThree;
    private ComputeBuffer passHistFour;

    private ComputeBuffer indexBuffer;
    private ComputeBuffer globalHistBuffer;
    private ComputeBuffer timingBuffer;

    OneSweepKeyValueHelper()
    {
        buffer_size = 0;
    }

    void Start()
    {
        CheckShader();

        UpdateGlobHistBuffer();
        UpdateIndexBuffer();
        UpdateTimingBuffer();
    }

    public void SortComputeBufferArray(int size, ComputeBuffer keys, ComputeBuffer payloads)
    {
        UpdateSize(size);
        UpdateSortBuffers(keys, payloads);
        ResetBuffers();

        compute.Dispatch(k_globalHist, globalHistThreadBlocks, 1, 1);
        if (size > partitionSize)
        {
            compute.Dispatch(k_scatterOne, size / partitionSize, 1, 1);
            compute.Dispatch(k_scatterTwo, size / partitionSize, 1, 1);
            compute.Dispatch(k_scatterThree, size / partitionSize, 1, 1);
            compute.Dispatch(k_scatterFour, size / partitionSize, 1, 1);
        }
        else
        {
            compute.Dispatch(k_scatterOne, 1, 1, 1);
            compute.Dispatch(k_scatterTwo, 1, 1, 1);
            compute.Dispatch(k_scatterThree, 1, 1, 1);
            compute.Dispatch(k_scatterFour, 1, 1, 1);
        }
    }

    private void CheckShader()
    {
        try
        {
            compute.FindKernel("InitOneSweepKeyValue");
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

        if (this.buffer_size < _size)
        {
            if (passHistBuffer != null)
                passHistBuffer.Dispose();
            if (passHistTwo != null)
                passHistTwo.Dispose();
            if (passHistThree != null)
                passHistThree.Dispose();
            if (passHistFour != null)
                passHistFour.Dispose();

            buffer_size = (int)(_size * 1.5);

            altBuffer = new ComputeBuffer(buffer_size, sizeof(uint));
            payloadAltBuffer = new ComputeBuffer(buffer_size, sizeof(uint));

            passHistBuffer = new ComputeBuffer(buffer_size / partitionSize * radix, sizeof(uint));
            passHistTwo = new ComputeBuffer(buffer_size / partitionSize * radix, sizeof(uint));
            passHistThree = new ComputeBuffer(buffer_size / partitionSize * radix, sizeof(uint));
            passHistFour = new ComputeBuffer(buffer_size / partitionSize * radix, sizeof(uint));
        }

        UpdatePassHistBuffer(_size);
    }

    private void UpdateSortBuffers(ComputeBuffer keys, ComputeBuffer payloads)
    {
        compute.SetBuffer(k_init, "b_sort", keys);
        compute.SetBuffer(k_globalHist, "b_sort", keys);

        compute.SetBuffer(k_scatterOne, "b_sort", keys);
        compute.SetBuffer(k_scatterOne, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterOne, "b_sortPayload", payloads);
        compute.SetBuffer(k_scatterOne, "b_altPayload", payloadAltBuffer);

        compute.SetBuffer(k_scatterTwo, "b_sort", keys);
        compute.SetBuffer(k_scatterTwo, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterTwo, "b_sortPayload", payloads);
        compute.SetBuffer(k_scatterTwo, "b_altPayload", payloadAltBuffer);

        compute.SetBuffer(k_scatterThree, "b_sort", keys);
        compute.SetBuffer(k_scatterThree, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterThree, "b_sortPayload", payloads);
        compute.SetBuffer(k_scatterThree, "b_altPayload", payloadAltBuffer);

        compute.SetBuffer(k_scatterFour, "b_sort", keys);
        compute.SetBuffer(k_scatterFour, "b_alt", altBuffer);
        compute.SetBuffer(k_scatterFour, "b_sortPayload", payloads);
        compute.SetBuffer(k_scatterFour, "b_altPayload", payloadAltBuffer);
    }

    private void UpdatePassHistBuffer(int _size)
    {
        //init
        compute.SetBuffer(k_init, "b_passHist", passHistBuffer);
        compute.SetBuffer(k_init, "b_passTwo", passHistTwo);
        compute.SetBuffer(k_init, "b_passThree", passHistThree);
        compute.SetBuffer(k_init, "b_passFour", passHistFour);

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

    private void ResetBuffers()
    {
        compute.Dispatch(k_init, 256, 1, 1);
    }


    private void OnDestroy()
    {
        if (altBuffer != null)
            altBuffer.Dispose();
        if (payloadAltBuffer != null)
            payloadAltBuffer.Dispose();

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
