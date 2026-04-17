using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public interface IRecPostprocess
    {
        InferenceResult[] RecPostProcess(OrtValue ortValue, float[] wh_ratio_list, float max_wh_ratio, string[] charList);
    }
}
