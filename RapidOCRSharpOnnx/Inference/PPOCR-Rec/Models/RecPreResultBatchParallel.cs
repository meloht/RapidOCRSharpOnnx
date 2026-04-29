using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models
{
    public record RecPreResultBatchParallel(float[] InputData, OrtValue InputVal, int BatchIndex,float MaxWhRatio, float[] WhRatioList);

}
