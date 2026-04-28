using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx
{
    public class OcrResult
    {
        public string TextBlocks { get; set; } = string.Empty;

        public ResultPerf<DetResult> DetResult { get; set; }

        public ResultPerf<ClsResult[]> ClsResult { get; set; }

        public ResultPerf<RecResult[]> RecResult { get; set; }

        public override string ToString()
        {
            return $"TextBlocks: {TextBlocks}{System.Environment.NewLine}{System.Environment.NewLine}DetPerf: {DetResult?.Perf}{System.Environment.NewLine}ClsPerf: {ClsResult?.Perf}{System.Environment.NewLine}RecPerf: {RecResult?.Perf}{System.Environment.NewLine}";
        }

    }
}
