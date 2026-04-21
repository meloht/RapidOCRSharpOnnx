using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public struct DetPreprocessData
    {
        public float[] Data { get; private set; }
        public long[] Dimensions { get; private set; }
        public float RatioH { get; set; }
        public float RatioW { get; set; }

        public int PaddingTop { get; set; }
        public int PaddingLeft { get; set; }

        public DetPreprocessData(float[] data, long[] dimensions)
        {
            Data = data;
            Dimensions = dimensions;
        }
    }
}
