using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx
{
    public class OcrBatchResult
    {
        public string ImagePath { get; set; }
        public string TextBlocks { get; set; } = string.Empty;

        public DetResult DetResult { get; set; }

        public ClsResult[] ClsResult { get; set; }

        public RecResult[] RecResult { get; set; }
    }
}
