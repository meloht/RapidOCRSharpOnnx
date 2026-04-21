using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public interface IDetPreprocess
    {
        DetPreprocessData Preprocess(Mat image, Mat resizedImg);
    }
}
