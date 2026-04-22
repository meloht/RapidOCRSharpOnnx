using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public class ClsResult
    {
        public string Label { get; set; }
        public float Score { get; set; }


        public ClsResult(string label, float score)
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
