using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.Config
{
    public static class ClsConfig
    {
        public static string LangType = "ch";
        public static string ModelType = "mobile";
        public static string OcrVersion = "PP-OCRv4";
        public static string ModelPath = "";
        public static string ModelDir = "";

        public static int[] ClsImageShape = [3, 48, 192];
        public static int ClsBatchNum = 6;
        public static float ClsThresh = 0.9f;
        public static string[] LabelList = ["0", "180"];
    }
}
