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
    public interface IOcrClassifier : IDisposable
    {
        ResultPerf<ClsResult[]> TextClassify(DisposableList<ImageIndex> imgList);

        ResultPerf<ClsResult[]> TextClassifySeq(DisposableList<ImageIndex> imgList);

        Task BatchParallelClsAsync(OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> recChannelWriter);

        Task BatchClsAsync(OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> nextChannelWriter);
    }
}
