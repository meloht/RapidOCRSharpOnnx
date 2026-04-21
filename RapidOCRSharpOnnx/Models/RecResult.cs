using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public struct RecResult
    {
        public string Label { get; set; }
        public float Score { get; set; }
        public WordInfo WordInfo { get; set; }

        public RecResult(string label, float score)
        {
            Label = label;
            Score = score;
        }
        public override string ToString()
        {
            return $"Label: {Label}, Score: {Score}";
        }
    }
}
