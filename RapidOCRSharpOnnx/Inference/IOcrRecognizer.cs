using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrRecognizer : IDisposable
    {
        ResultPerf<RecResult[]> TextRecognize(DisposableList<ImageIndex> imgList);

        ResultPerf<RecResult[]> TextRecognizeSeq(DisposableList<ImageIndex> imgList);

        void BatchRecAsync(OcrBatchResult batchResult);
    }
}
