using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrDetector : IDisposable
    {
        ResultPerf<DetResult> TextDetect(Mat image);

        Task BatchDetectAsync(List<string> listImg, ChannelWriter<OcrBatchResult> nextChannelWriter, OcrBatchResult[] batchResults);
    }
}
