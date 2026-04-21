using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public record WordResult(List<string> words, List<float> confs, List<Point2f[]> boxes);
}
