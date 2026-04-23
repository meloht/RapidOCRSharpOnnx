using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public interface IRecPreprocess
    {
        int ResizeNormImg(Mat img, int idx, float[] inputData, float max_wh_ratio);

        void PreprocessBatchAsync(DisposableList<Mat> ImgCropList, DeviceType deviceType, OcrBatchResult batchResult, ChannelWriter<RecPreResultBatch> writer);
    }
}
