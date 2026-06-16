using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Configurations
{
    internal static class PPOCRv6DefaultConfig
    {
        public static float[] Mean = [0.485f, 0.456f, 0.406f];
        public static float[] Std = [0.229f, 0.224f, 0.225f];

        public static float Thresh = 0.2f;
        public static float BoxThresh = 0.45f;
        public static int MaxCandidates = 3000;
        public static float UnclipRatio = 1.4f;
        public static bool UseDilation = false;

        public static void SetDefault(DetectorConfig config)
        {
            config.Mean = Mean;
            config.Std = Std;
            config.BoxThresh = BoxThresh;
            config.MaxCandidates = MaxCandidates;
            config.Thresh = Thresh;
            config.UnclipRatio = UnclipRatio;
            config.UseDilation = UseDilation;
        }
    }
}
