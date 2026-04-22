using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public interface IDetPostprocess
    {
        DetResult PostProcess(Mat image, OrtValue output);
    }
}
