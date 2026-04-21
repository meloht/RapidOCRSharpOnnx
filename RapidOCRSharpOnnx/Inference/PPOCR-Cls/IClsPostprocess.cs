using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public interface IClsPostprocess
    {
        void ClsPostProcess(OrtValue ortValue, int ij, Mat[] imgList, ClsResult[] cls_res);
    }
}
