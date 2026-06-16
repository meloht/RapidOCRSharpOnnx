using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Configurations
{
    internal static class PPOCRv5DefaultConfig
    {
        public static float[] Mean = [0f, 0f, 0f];
        public static float[] Std  = [1f, 1f, 1f];

        public static float Thresh  = 0.3f;
        public static float BoxThresh  = 0.5f;
        public static int MaxCandidates  = 1000;
        public static float UnclipRatio  = 1.6f;
        public static bool UseDilation  = true;

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
