using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models
{
    public record RecPreResultBatch(float[] InputData,int Index, float MaxWhRatio, float WhRatio,int ImgWidth);
}
