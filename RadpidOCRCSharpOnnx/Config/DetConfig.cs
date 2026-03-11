using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.Config
{
    public static class DetConfig
    {
        public static string LangType = "ch";
        public static string ModelType = "mobile";
        public static string OcrVersion = "PP-OCRv4";
        public static string ModelPath = "";
        public static string ModelDir = "";
        public static int LimitSideLen = 736;
        public static string LimitType = "min";
        public static float[] Std = [0.5f, 0.5f, 0.5f];
        public static float[] Mean = [0.5f, 0.5f, 0.5f];
        public static float Thresh = 0.3f;
        public static float BoxThresh = 0.5f;
        public static int MaxCandidates = 1000;
        public static float UnclipRatio = 1.6f;
        public static bool UseDilation = true;
        public static string ScoreMode = "fast";
    }
}
