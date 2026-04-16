using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public struct InferenceResult
    {
        public string Label { get; set; }
        public float Score { get; set; }

        public InferenceResult(string label, float score)
        {
            Label = label;
            Score = score;
        }
    }
}
