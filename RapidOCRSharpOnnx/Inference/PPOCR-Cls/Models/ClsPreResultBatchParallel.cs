using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models
{
    public record ClsPreResultBatchParallel(float[] InputData, OrtValue InputVal,int BatchIndex);

}
