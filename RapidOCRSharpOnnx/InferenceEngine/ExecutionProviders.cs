using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public enum ExecutionProviders
    {
        /// <summary>
        /// CPUExecutionProvider
        /// </summary>
        CPU_EP,
        /// <summary>
        /// CUDAExecutionProvider
        /// </summary>
        CUDA_EP,
        /// <summary>
        /// DmlExecutionProvider
        /// </summary>
        DirectML_EP,
      
        /// <summary>
        /// CoreMLExecutionProvider
        /// </summary>
        CoreML_EP
    }

    public static class EpNames
    {
        public const string CPUExecutionProvider = "CPUExecutionProvider";
        public const string CUDAExecutionProvider = "CUDAExecutionProvider";
        public const string DmlExecutionProvider = "DmlExecutionProvider";
        public const string CoreMLExecutionProvider = "CoreMLExecutionProvider";
    }


   
}
