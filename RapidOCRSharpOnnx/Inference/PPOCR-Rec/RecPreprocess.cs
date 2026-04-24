using OpenCvSharp;

using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class RecPreprocess : PreprocessBatchCore<ImageIndex, RecPreResultBatch>, IRecPreprocess
    {
        private RecognizerConfig _recConfig;
        public RecPreprocess(RecognizerConfig recConfig)
        {
            _recConfig = recConfig;
        }
        public int ResizeNormImg(Mat img, int idx, float[] inputData, int img_width, int max_wh_ratio)
        {
            // 获取原图尺寸和通道数
            int h = img.Height;
            int w = img.Width;
            int channels = img.Channels();
            int img_c = _recConfig.RecImgShape[0];
            int img_h = _recConfig.RecImgShape[1];

            if (img_c != channels)
                throw new ArgumentException($"The count of image channels does not match：expect {img_c}，actual {channels}");

            // 计算缩放后的宽度（保持宽高比，但不超过目标宽度）
            float ratio = (float)w / h;
            int estimatedWidth = (int)Math.Round(Math.Ceiling(img_h * ratio), 0);

            int resized_w = estimatedWidth > max_wh_ratio ? max_wh_ratio : estimatedWidth;

            // 缩放图像到 (resized_w, img_h)
            using Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h));

            for (int i = 0; i < img_c; i++)
            {
                for (int j = 0; j < img_h; j++)
                {
                    for (int k = 0; k < img_width; k++)
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


        public void PreprocessBatchAsync(DisposableList<ImageIndex> imgCropList, DeviceType deviceType, ChannelWriter<RecPreResultBatch> writer)
        {

            PreprocessBatchBaseAsync(imgCropList, deviceType, writer, PreprocessChannel);
        }
        protected RecPreResultBatch PreprocessChannel(ImageIndex batchImage)
        {
            return PreprocessSeq(batchImage);
        }

        public RecPreResultBatch PreprocessSeq(ImageIndex batchImage)
        {
            Mat img = batchImage.Image;
            int img_c = _recConfig.RecImgShape[0];
            int img_h = _recConfig.RecImgShape[1];
            int img_w = _recConfig.RecImgShape[2];
            float max_wh_ratio = (float)img_w / (float)img_h;
            float wh_ratio = (float)img.Width / (float)img.Height;
            max_wh_ratio = Math.Max(max_wh_ratio, wh_ratio);

            int img_width = (int)Math.Round(img_h * max_wh_ratio, 0);
            int tensorLength = img_c * img_h * img_width;

            float[] inputData = new float[tensorLength];
            ResizeNormImg(img, 0, inputData, img_width, img_width);

            return new RecPreResultBatch(inputData, batchImage.Index, max_wh_ratio, wh_ratio, img_width);
        }
    }
}
