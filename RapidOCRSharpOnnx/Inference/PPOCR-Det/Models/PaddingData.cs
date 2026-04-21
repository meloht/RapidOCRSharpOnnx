using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det.Models
{
    internal struct PaddingData
    {
        public int Top { get; set; }
        public int Left { get; set; }

        public PaddingData(int top, int left)
        {
            Top = top;
            Left = left;
        }
    }
}
