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
        /// <summary>
        /// millisecond
        /// </summary>
        public long DetElapsedTime { get; set; }
        /// <summary>
        /// millisecond
        /// </summary>
        public long ClsElapsedTime { get; set; }
        /// <summary>
        /// millisecond
        /// </summary>
        public long RecElapsedTime { get; set; }

        public override string ToString()
        {
            return $"{TextBlocks}{System.Environment.NewLine}{System.Environment.NewLine}TotalTime: {DetElapsedTime + ClsElapsedTime + RecElapsedTime}ms DetInfer:{DetElapsedTime}ms ClsInfer:{ClsElapsedTime}ms RecInfer:{RecElapsedTime}ms{System.Environment.NewLine}";
        }
    }
}
