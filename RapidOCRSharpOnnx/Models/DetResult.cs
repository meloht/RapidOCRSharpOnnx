using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetResult
    {
        public DetBoxItem[] DetItems { get; set; }
        public DisposableList<Mat> ImgCropList { get; set; }
        public float RatioH { get; set; }
        public float RatioW { get; set; }

        public int PaddingTop { get; set; }
        public int PaddingLeft { get; set; }

        public DetResult(DetBoxItem[] detItems)
        {
            DetItems = detItems;
        }
    }
}
