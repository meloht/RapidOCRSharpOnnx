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
        int ResizeNormImg(Mat img, int idx, float[] inputData);

        void PreprocessBatchAsync(DisposableList<ImageIndex> ImgCropList, DeviceType deviceType, ChannelWriter<ClsPreResultBatch> writer);

        int[] GetClsImageShape();

    }
}
