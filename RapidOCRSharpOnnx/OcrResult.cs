using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx
{
    public class OcrResult
    {
        public string TextBlocks { get; set; }

        public ResultPerf<DetResult> DetResults { get; set; }

        public ResultPerf<ClsResult[]> ClsResults { get; set; }

        public ResultPerf<RecResult[]> RecResults { get; set; }

        public override string ToString()
        {
            return $"TextBlocks: {TextBlocks} DetPerf: {DetResults.Perf}, ClsPerf: {ClsResults.Perf}, RecPerf: {RecResults.Perf}";
        }

    }
}
