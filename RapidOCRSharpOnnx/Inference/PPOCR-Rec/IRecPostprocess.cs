using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public interface IRecPostprocess
    {
        RecResult[] RecPostProcess(OrtValue ortValue, float[] wh_ratio_list, float max_wh_ratio, string[] charList);
    }
}
