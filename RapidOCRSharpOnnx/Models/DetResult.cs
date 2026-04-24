using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetResult
    {
        public DetBoxItem[] DetItems { get; set; }
        public DisposableList<ImageIndex> ImgCropList { get; set; }

        public ResizeData ResizeData { get; set; }

        public DetResult(DetBoxItem[] detItems)
        {
            DetItems = detItems;
        }
    }
}
