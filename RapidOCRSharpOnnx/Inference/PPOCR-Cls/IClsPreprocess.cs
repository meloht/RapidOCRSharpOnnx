using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public interface IClsPreprocess
    {
        int ResizeNormImg(Mat img, int idx, float[] inputData);

        void PreprocessBatchAsync(DisposableList<Mat> ImgCropList, DeviceType deviceType, OcrBatchResult batchResult, ChannelWriter<ClsPreResultBatch> writer);

        int[] GetClsImageShape();

    }
}
