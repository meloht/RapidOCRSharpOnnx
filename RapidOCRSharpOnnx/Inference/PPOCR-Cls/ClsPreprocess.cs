using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public class ClsPreprocess : PreprocessBatchCore<Mat, OcrBatchResult, ClsPreResultBatch>, IClsPreprocess
    {
        private static readonly int[] ClsImageShapev4 = [3, 48, 192];
        private static readonly int[] ClsImageShapev5 = [3, 80, 160];

        private OcrConfig _ocrConfig;
        protected readonly int[] _clsImageShape;
        public ClsPreprocess(OcrConfig ocrConfig)
        {
            _ocrConfig = ocrConfig;
            if (_ocrConfig.ClassifierConfig.OCRVersion == OCRVersion.PPOCRV5)
            {
                _clsImageShape = ClsImageShapev5;
            }
            else
            {
                _clsImageShape = ClsImageShapev4;
            }
        }

        public int[] GetClsImageShape()
        {
            return _clsImageShape;
        }

        public int ResizeNormImg(Mat img, int idx, float[] inputData)
        {
            // 获取原图尺寸和通道数
            int h = img.Height;
            int w = img.Width;
            int channels = img.Channels();
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];

            if (img_c != channels)
                throw new ArgumentException($"The count of image channels does not match：expect {img_c}，actual {channels}");

            // 计算缩放后的宽度（保持宽高比，但不超过目标宽度）
            float ratio = (float)w / h;
            double estimatedWidth = Math.Ceiling(img_h * ratio);

            int resized_w = estimatedWidth > img_w ? img_w : (int)estimatedWidth;

            // 缩放图像到 (resized_w, img_h)
            using Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h), interpolation: InterpolationFlags.Linear);

            for (int i = 0; i < img_c; i++)
            {
                for (int j = 0; j < img_h; j++)
                {
                    for (int k = 0; k < img_w; k++)
                    {
                        if (k < resized_w)
                        {
                            var val = (float)resized.At<Vec3b>(j, k)[i];
                            val = (val / 255.0f) * 2f - 1f;
                            inputData[idx++] = val;
                        }
                        else
                        {
                            inputData[idx++] = 0.0f;
                        }
                    }
                }
            }
            return idx;
        }

        public void PreprocessBatchAsync(DisposableList<Mat> ImgCropList, DeviceType deviceType, OcrBatchResult batchResult, ChannelWriter<ClsPreResultBatch> writer)
        {
            PreprocessBatchBaseAsync(ImgCropList, deviceType, writer, batchResult, PreprocessChannel);
        }
        protected ClsPreResultBatch PreprocessChannel(Mat img, OcrBatchResult batchResult)
        {
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];
            float[] inputData = new float[img_c * img_h * img_w];
            ResizeNormImg(img, 0, inputData);
            return new ClsPreResultBatch(batchResult, inputData, img);
        }


    }
}
