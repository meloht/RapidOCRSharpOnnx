using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class RecPreprocess : PreprocessBatchCore<ImageIndex, object, RecPreResultBatch>, IRecPreprocess
    {
        private RecognizerConfig _recConfig;
        public RecPreprocess(RecognizerConfig recConfig)
        {
            _recConfig = recConfig;
        }
        public void ResizeNormImg(Mat img, int idx, float[] inputData, int img_width, int max_wh_ratio)
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

            unsafe
            {
                fixed (float* dst = inputData)
                {
                    ConvertToNormImg(resized_w, idx, img_c, img_h, img_width, resized, dst);
                }

            }


        }

        private RecPreResultBatchParallel ResizeNormImg(ImageIndex[] items, int batchIndex)
        {
            int img_c = _recConfig.RecImgShape[0];
            int img_h = _recConfig.RecImgShape[1];
            int img_w = _recConfig.RecImgShape[2];

            int batchSize = items.Length;
            float[] wh_ratio_list = new float[batchSize];

            float config_wh_ratio = (float)img_w / (float)img_h;
            float[] max_wh_ratio_list = new float[batchSize];
            float max_wh_ratio = config_wh_ratio;

            for (int i = 0; i < batchSize; i++)
            {
                float wh_ratio = (float)items[i].Image.Width / (float)items[i].Image.Height;

                wh_ratio_list[i] = wh_ratio;
                max_wh_ratio_list[i] = Math.Max(config_wh_ratio, wh_ratio);
                max_wh_ratio = Math.Max(max_wh_ratio, max_wh_ratio_list[i]);
            }

            int img_width = (int)Math.Round(img_h * max_wh_ratio, 0);
            int tensorLength = img_c * img_h * img_width * batchSize;
            float[] batchData = ArrayPool<float>.Shared.Rent(tensorLength);

            Parallel.For(0, items.Length, index =>
            {
                int img_max_width = (int)Math.Round(img_h * max_wh_ratio_list[index], 0);
                ResizeNormImg(items[index].Image, index, batchData, img_width, img_max_width);
            });
            var ort = OrtValue.CreateTensorValueFromMemory(batchData, [batchSize, img_c, img_h, img_width]);
            return new RecPreResultBatchParallel(batchData, ort, batchIndex, max_wh_ratio, wh_ratio_list);
        }
        public void PreprocessBatchParallelAsync(DisposableList<ImageIndex> imgCropList, ChannelWriter<RecPreResultBatchParallel> writer)
        {
            var batchList = imgCropList.Chunk(_recConfig.RecBatchNum).ToArray();
            int count = batchList.Length;

            int[] dict = new int[count];
            int batchIndex = 0;
            for (int i = 0; i < count; i++)
            {
                dict[i] = batchIndex;
                batchIndex += batchList[i].Length;
            }

            Parallel.For(0, count, index =>
            {
                var items = batchList[index];
                var res = ResizeNormImg(items, dict[index]);
                writer.TryWrite(res);
            });
            writer.Complete();
        }

        public async Task PreprocessBatchAsync(DisposableList<ImageIndex> imgCropList, DeviceType deviceType, ChannelWriter<RecPreResultBatch> writer)
        {
            await PreprocessBatchBaseAsync(imgCropList, deviceType, null, writer, PreprocessChannel);
        }
        protected RecPreResultBatch PreprocessChannel(ImageIndex batchImage, object t2)
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

            float[] inputData = ArrayPool<float>.Shared.Rent(tensorLength);
            ResizeNormImg(img, 0, inputData, img_width, img_width);

            return new RecPreResultBatch(inputData, batchImage.Index, max_wh_ratio, wh_ratio, img_width);
        }
    }
}
