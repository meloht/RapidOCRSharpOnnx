using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public struct DataTensorDimensions
    {
        public float[] Data { get; private set; }
        public long[] Dimensions { get; private set; }

        public DataTensorDimensions(float[] data, long[] dimensions)
        {
            Data = data;
            Dimensions = dimensions;
        }
    }
}
