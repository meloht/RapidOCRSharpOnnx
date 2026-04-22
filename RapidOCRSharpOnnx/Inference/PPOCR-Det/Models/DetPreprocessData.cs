using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetPreprocessData
    {
        public float[] Data { get; private set; }
        public long[] Dimensions { get; private set; }
        public ResizeData ResizeData { get; set; }

        public DetPreprocessData(float[] data, long[] dimensions)
        {
            Data = data;
            Dimensions = dimensions;
        }
    }
}
