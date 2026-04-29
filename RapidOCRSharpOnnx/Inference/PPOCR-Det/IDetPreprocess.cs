using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Providers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public interface IDetPreprocess
    {
        DetPreprocessData Preprocess(Mat image, Mat resizedImg);

        Task PreprocessBatchAsync(List<ImagePathIndex> listImg, DeviceType deviceType, ChannelWriter<DetPreResultBatch> writer);
    }
}
