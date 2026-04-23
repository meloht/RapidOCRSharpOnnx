using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference
{
    public interface IOcrClassifier : IDisposable
    {
        ResultPerf<ClsResult[]> TextClassify(DisposableList<Mat> imgList);

        void BatchClsAsync(OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> nextChannelWriter);
    }
}
