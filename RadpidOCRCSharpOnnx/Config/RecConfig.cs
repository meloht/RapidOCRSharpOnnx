using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.Config
{
    public static class RecConfig
    {
        public static string LangType  = "ch";
        public static string ModelType = "mobile";
        public static string OcrVersion = "PP-OCRv4";
        public static string ModelPath = "";
        public static string ModelDir = "";

        public static string RecKeysPath = "";
        public static int[] RecImgShape = [3, 48, 320];
        public static int RecBatchNum = 6;
    }
}
