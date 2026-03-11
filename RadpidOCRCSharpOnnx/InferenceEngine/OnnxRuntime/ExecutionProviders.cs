using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine.OnnxRuntime
{
    public enum ExecutionProviders
    {
        /// <summary>
        /// CPUExecutionProvider
        /// </summary>
        [Description("CPUExecutionProvider")]
        CPU_EP,
        /// <summary>
        /// CUDAExecutionProvider
        /// </summary>
        [Description("CUDAExecutionProvider")]
        CUDA_EP,
        /// <summary>
        /// DmlExecutionProvider
        /// </summary>
        [Description("DmlExecutionProvider")]
        DIRECTML_EP,
        /// <summary>
        /// CANNExecutionProvider
        /// </summary>
        [Description("CANNExecutionProvider")]
        CANN_EP,
        /// <summary>
        /// CoreMLExecutionProvider
        /// </summary>
        [Description("CoreMLExecutionProvider")]
        COREML_EP
    }

    public static class EnumExtensions
    {
        public static string GetDescription(this Enum value)
        {
            var field = value.GetType().GetField(value.ToString());
            var attribute = (DescriptionAttribute)Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute));
            return attribute == null ? value.ToString() : attribute.Description;
        }
    }
}
