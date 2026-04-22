using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det.Models
{
    public class ResizeData
    {
        public float RatioH { get; set; }
        public float RatioW { get; set; }

        public int PaddingTop { get; set; }
        public int PaddingLeft { get; set; }
    }
}
