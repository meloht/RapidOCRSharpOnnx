using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public class RecResult
    {
        public string Label { get; set; }
        public float Score { get; set; }

        public float LineTxtLen { get; set; }

        public List<int> ValidCols { get; set; }

        public List<float> ConfList { get; set; }

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
