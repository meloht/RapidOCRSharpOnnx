using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det.Models
{
    public class DetBatchResult
    {
        public string ImagePath { get; set; }

        public DetResult Data { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }
    }

   // public record DetBatchResult(DetPreprocessData PreResult, Mat resizedImg, string ImagePath);
}
