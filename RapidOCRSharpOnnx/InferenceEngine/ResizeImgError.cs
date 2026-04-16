using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public class ResizeImgError : Exception
    {
        public ResizeImgError() : base() { }
        public ResizeImgError(string message) : base(message) { }
        public ResizeImgError(string message, Exception innerException)
            : base(message, innerException) { }
    }
}
