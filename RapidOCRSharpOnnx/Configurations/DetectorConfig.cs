using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class DetectorConfig
    {
        public required string ModelPath { get; set; }

        public int LimitSideLen { get; set; } = 736;//736 or 960
        public LimitType LimitType { get; set; } = LimitType.Min;
        public float[] Mean { get; set; } = [0.485f, 0.456f, 0.406f]; 
        public float[] Std { get; set; } = [0.229f, 0.224f, 0.225f];
       
        public float Thresh { get; set; } = 0.3f;
        public float BoxThresh { get; set; } = 0.6f;
        public int MaxCandidates { get; set; } = 3000;
        public float UnclipRatio { get; set; } = 1.5f;
        public bool UseDilation { get; set; } = false;
        public ScoreMode ScoreMode { get; set; } = ScoreMode.FAST;


    }
}
