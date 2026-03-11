using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine.OnnxRuntime
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
        DIRECTML_EP,
        /// <summary>
        /// CANNExecutionProvider
        /// </summary>
        CANN_EP,
        /// <summary>
        /// CoreMLExecutionProvider
        /// </summary>
        COREML_EP
    }
}
