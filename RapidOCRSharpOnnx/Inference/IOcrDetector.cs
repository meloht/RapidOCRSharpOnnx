using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrDetector : IDisposable
    {
        ResultPerf<DetResult> TextDetect(Mat image);
    }
}
