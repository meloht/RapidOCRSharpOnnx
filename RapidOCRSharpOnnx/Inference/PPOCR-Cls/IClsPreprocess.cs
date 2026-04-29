using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
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
        void ResizeNormImg(Mat img, int idx, Mat resized, float[] inputData);

        void ResizeNormImg(Mat img, Mat resized, FixedBuffer buffer);


        void PreprocessBatchParallelAsync(DisposableList<ImageIndex> imgCropList, int inputShapeSize, ChannelWriter<ClsPreResultBatchParallel> writer);

        Task PreprocessBatchAsync(DisposableList<ImageIndex> ImgCropList, MatBufferPool matBuffer, DeviceType deviceType, ChannelWriter<ClsPreResultBatch> writer);

        int[] GetClsImageShape();

    }
}
