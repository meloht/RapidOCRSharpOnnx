using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public enum IntelDeviceType
    {
        /// <summary>
        /// Intel® CPUs
        /// </summary>
        CPU,
        /// <summary>
        /// Intel integrated GPU or discrete GPU
        /// </summary>
        GPU,
        /// <summary>
        /// Specific GPU when multiple GPUs are available, GPU.0 Intel® Integrated Graphics
        /// </summary>
        GPU0,
        /// <summary>
        /// Specific GPU when multiple GPUs are available, GPU.1 Intel® Discrete Graphics
        /// </summary>
        GPU1,
        /// <summary>
        /// Intel® Neural Processor Unit
        /// </summary>
        NPU
    }
}
