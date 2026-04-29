using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class RecognizerConfig
    {
        public required string ModelPath { get; set; }
        public required LangRec LangRec { get; set; }
        public int[] RecImgShape = [3, 48, 320];
        public int RecBatchNum = 6;
       

    }
}
