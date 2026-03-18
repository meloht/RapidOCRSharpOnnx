using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.Config
{
    public static class GlobalConfig
    {
        public static float TextScore = 0.5f;
        public static bool UseDet = true;
        public static bool UseCls = true;
        public static bool UseRec = true;
        public static int MinHeight = 30;
        public static int WidthHeightRatio = 8;
        public static int MaxSideLen = 2000;
        public static int MinSideLen = 30;
        public static bool ReturnWordBox = false;
        public static bool ReturnSingleCharBox = false;
        public static string FontPath = string.Empty;
    }
}
