using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Config
{
    public static class DetConfig
    {
        public static string LangType = "en";
        public static string ModelType = "mobile";
        public static string OcrVersion = "PP-OCRv5";
        public static string ModelPath = @"D:\code\RapidOCR-3.7.0\python\rapidocr\models\ch_PP-OCRv5_server_det.onnx";
        public static string ModelDir = "";
        public static int LimitSideLen = 736;
        public static LimitType LimitType = LimitType.Min;
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
