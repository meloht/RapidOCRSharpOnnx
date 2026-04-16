using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public class ONNXRuntimeError : Exception
    {
        public ONNXRuntimeError() : base() { }
        public ONNXRuntimeError(string message) : base(message) { }
        public ONNXRuntimeError(string message, Exception innerException)
            : base(message, innerException) { }
    }
}
