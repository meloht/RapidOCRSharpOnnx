using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class ClassifierConfig
    {
        public required OCRVersion OCRVersion { get; set; }
        public required string ModelPath { get; set; }
        public int ClsBatchNum { get; set; } = 6;
        public float ClsThresh { get; set; } = 0.9f;
        public string[] LabelList { get; set; } = ["0", "180"];


    }
}
