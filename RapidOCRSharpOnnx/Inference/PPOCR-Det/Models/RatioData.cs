using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det.Models
{
    internal struct RatioData
    {
        public float RatioW { get; set; }
        public float RatioH { get; set; }

        public RatioData(float ratioH, float ratioW)
        {
            RatioW = ratioW;
            RatioH = ratioH;
        }
    }
}
