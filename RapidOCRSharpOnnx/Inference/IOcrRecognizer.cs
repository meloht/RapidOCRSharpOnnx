using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrRecognizer : IDisposable
    {
        RecResult[] TextRecognize(Mat[] imgList);
    }
}
