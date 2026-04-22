using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det.Models
{

    public record DetPreResultBatch(DetPreprocessData PreResult, Mat resizedImg, string ImagePath);
}
