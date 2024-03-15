/******************************************************************************
 * OneSweep Implementation Toy Demo
 *
 * SPDX-License-Identifier: MIT
 * Author:  Thomas Smith 3/14/2024
 * 
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 ******************************************************************************/
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;

public class OneSweep : MonoBehaviour
{
    [Range(k_minSize, k_maxSize)]
    public int m_sizeExponent;

    [SerializeField]
    private ComputeShader m_compute;

    private ComputeBuffer m_sortBuffer;
    private ComputeBuffer m_altBuffer;
    private ComputeBuffer m_globalHistBuffer;
    private ComputeBuffer m_indexBuffer;
    private ComputeBuffer m_passHistBuffer;
    private ComputeBuffer m_errCountBuffer;

    private const int k_minSize = 15;
    private const int k_maxSize = 27;

    private const int m_initOneSweepKernel = 0;
    private const int m_globalHistKernel = 1;
    private const int m_scanKernel = 2;
    private const int m_digitBinPassKernel = 3;
    private const int m_initRandomKernel = 4;
    private const int m_validationKernel = 5;

    private const int k_radixPasses = 4;
    private const int k_radix = 256;
    private const int k_partitionSize = 3840;
    private const string k_computeShaderString = "OneSweep";

    private int m_size = 0;
    private int m_threadBlocks = 0;

    private void Start()
    {
        CheckShader();
        m_size = 1 << 15;
        m_threadBlocks = divRoundUp(m_size, k_partitionSize);
        UpdateSize();
        UpdateGlobHistBuffer();
        UpdateIndexBuffer();
        UpdateErrorBuffer();

        Debug.Log(k_computeShaderString + ": init Complete.");
        Debug.Log("Press space to run and test OneSweep at the current size.");
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (m_size != (1 << m_sizeExponent))
            {
                m_size = 1 << m_sizeExponent;
                m_threadBlocks = divRoundUp(m_size, k_partitionSize);
                UpdateSize();
            }
            ValidateSort();
        }
    }

    private void CheckShader()
    {
        try
        {
            m_compute.FindKernel("Init" + k_computeShaderString);
        }
        catch
        {
            Debug.LogError("Kernel(s) not found, most likely you do not have the correct compute shader attached to the game object");
            Debug.LogError("The correct compute shader is" + k_computeShaderString + ". Exit play mode and attatch to the gameobject, then retry.");
            Debug.LogError("Destroying this object.");
            Destroy(this);
        }
    }

    private void UpdateSize()
    {
        m_compute.SetInt("e_numKeys", m_size);
        m_compute.SetInt("e_threadBlocks", m_threadBlocks);
        UpdateSortBuffers();
        UpdatePassHistBuffer();
    }

    private void UpdateSortBuffers()
    {
        if (m_sortBuffer != null)
            m_sortBuffer.Dispose();
        if (m_altBuffer != null)
            m_altBuffer.Dispose();

        m_sortBuffer = new ComputeBuffer(m_size, sizeof(uint));
        m_altBuffer = new ComputeBuffer(m_size, sizeof(uint));
    }

    private void UpdatePassHistBuffer()
    {
        if (m_passHistBuffer != null)
            m_passHistBuffer.Dispose();

        m_passHistBuffer = new ComputeBuffer(m_threadBlocks * k_radix * k_radixPasses, sizeof(uint));
    }

    private void UpdateGlobHistBuffer()
    {
        if (m_globalHistBuffer != null)
            m_globalHistBuffer.Dispose();
        m_globalHistBuffer = new ComputeBuffer(k_radixPasses * k_radix, sizeof(uint));
    }

    private void UpdateIndexBuffer()
    {
        if (m_indexBuffer != null)
            m_indexBuffer.Dispose();
        m_indexBuffer = new ComputeBuffer(k_radixPasses, sizeof(uint));
    }

    private void UpdateErrorBuffer()
    {
        if (m_errCountBuffer != null)
            m_errCountBuffer.Dispose();
        m_errCountBuffer = new ComputeBuffer(1, sizeof(uint));
    }
    private void SetStaticBuffers()
    {
        //Input
        m_compute.SetBuffer(m_initRandomKernel, "b_sort", m_sortBuffer);

        //Init
        m_compute.SetBuffer(m_initOneSweepKernel, "b_passHist", m_passHistBuffer);
        m_compute.SetBuffer(m_initOneSweepKernel, "b_globalHist", m_globalHistBuffer);
        m_compute.SetBuffer(m_initOneSweepKernel, "b_index", m_indexBuffer);

        //GlobalHist
        m_compute.SetBuffer(m_globalHistKernel, "b_sort", m_sortBuffer);
        m_compute.SetBuffer(m_globalHistKernel, "b_globalHist", m_globalHistBuffer);

        //Scan
        m_compute.SetBuffer(m_scanKernel, "b_globalHist", m_globalHistBuffer);
        m_compute.SetBuffer(m_scanKernel, "b_passHist", m_passHistBuffer);

        //DigitBinningPass
        m_compute.SetBuffer(m_digitBinPassKernel, "b_passHist", m_passHistBuffer);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_globalHist", m_globalHistBuffer);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_index", m_indexBuffer);

        //Validate
        m_compute.SetBuffer(m_validationKernel, "b_sort", m_sortBuffer);
        m_compute.SetBuffer(m_validationKernel, "b_errorCount", m_errCountBuffer);
    }

    private void DispatchKernels()
    {
        SetStaticBuffers();
        m_compute.SetInt("e_seed", (int)(Time.realtimeSinceStartup * 100000.0f));
        m_compute.Dispatch(m_initRandomKernel, 256, 1, 1);

        m_compute.Dispatch(m_initOneSweepKernel, 256, 1, 1);
        m_compute.Dispatch(m_globalHistKernel, m_threadBlocks, 1, 1);
        
        m_compute.Dispatch(m_scanKernel, k_radixPasses, 1, 1);

        m_compute.SetInt("e_radixShift", 0);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_sort", m_sortBuffer);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_alt", m_altBuffer);
        m_compute.Dispatch(m_digitBinPassKernel, m_threadBlocks, 1, 1);

        m_compute.SetInt("e_radixShift", 8);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_sort", m_altBuffer);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_alt", m_sortBuffer);
        m_compute.Dispatch(m_digitBinPassKernel, m_threadBlocks, 1, 1);

        m_compute.SetInt("e_radixShift", 16);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_sort", m_sortBuffer);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_alt", m_altBuffer);
        m_compute.Dispatch(m_digitBinPassKernel, m_threadBlocks, 1, 1);

        m_compute.SetInt("e_radixShift", 24);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_sort", m_altBuffer);
        m_compute.SetBuffer(m_digitBinPassKernel, "b_alt", m_sortBuffer);
        m_compute.Dispatch(m_digitBinPassKernel, m_threadBlocks, 1, 1);
    }

    private void ValidateSort()
    {
        DispatchKernels();
        uint[] errCount = new uint[1] { 0 };
        m_errCountBuffer.SetData(errCount);
        m_compute.Dispatch(m_validationKernel, 256, 1, 1);
        m_errCountBuffer.GetData(errCount);

        if(errCount[0] == 0)
            Debug.Log("OneSweep passed test at size " + m_size + ".");
        else
            Debug.LogError("OneSweep failed test at size " + m_size + " with " + errCount[0] + " errors.");
    }

    static int divRoundUp(int x, int y)
    {
        return (x + y - 1) / y;
    }

    private void OnDestroy()
    {
        if (m_sortBuffer != null)
            m_sortBuffer.Dispose();
        if (m_altBuffer != null)
            m_altBuffer.Dispose();
        if (m_globalHistBuffer != null)
            m_globalHistBuffer.Dispose();
        if (m_indexBuffer != null)
            m_indexBuffer.Dispose();
        if (m_passHistBuffer != null)
            m_passHistBuffer.Dispose();
        if (m_errCountBuffer != null)
            m_errCountBuffer.Dispose();
    }
}