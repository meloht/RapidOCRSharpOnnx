using OpenCvSharp;
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
        ResultPerf<RecResult[]> TextRecognize(DisposableList<Mat> imgList);

        void BatchRecAsync(OcrBatchResult batchResult);
    }
}
