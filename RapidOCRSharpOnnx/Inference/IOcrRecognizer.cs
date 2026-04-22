using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrRecognizer : IDisposable
    {
        ResultPerf<RecResult[]> TextRecognize(DisposableList<Mat> imgList);
    }
}
