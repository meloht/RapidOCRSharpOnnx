using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class DetectorConfig
    {
        public required string ModelPath { get; set; }

        public int LimitSideLen { get; set; } = 736;
        public LimitType LimitType { get; set; } = LimitType.Min;
        public float[] Std { get; set; } = new float[] { 0.229f, 0.224f, 0.225f };
        public float[] Mean { get; set; } = new float[] { 0.485f, 0.456f, 0.406f };
        public float Thresh { get; set; } = 0.3f;
        public float BoxThresh { get; set; } = 0.5f;
        public int MaxCandidates { get; set; } = 1000;
        public float UnclipRatio { get; set; } = 1.6f;
        public bool UseDilation { get; set; } = true;
        public ScoreMode ScoreMode { get; set; } = ScoreMode.FAST;




    }
}
