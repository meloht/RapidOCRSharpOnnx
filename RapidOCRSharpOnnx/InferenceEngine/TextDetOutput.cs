using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    
    public struct TextDetOutput
    {
        public List<Point2f[]> Boxs { get; }
        public List<float> Scores { get; }

        public long Elapse { get; }

        public TextDetOutput(List<Point2f[]> boxs, List<float> scores, long elapse)
        {
            Boxs = boxs;
            Scores = scores;
            Elapse = elapse;
        }
    }
}
