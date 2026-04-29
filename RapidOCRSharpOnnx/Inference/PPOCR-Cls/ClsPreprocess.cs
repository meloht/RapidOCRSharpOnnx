using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Flann;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using System.Threading.Channels;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public class ClsPreprocess : PreprocessBatchCore<ImageIndex, MatBufferPool, ClsPreResultBatch>, IClsPreprocess
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

        public void ResizeNormImg(Mat img, int idx, Mat resized, float[] inputData)
        {
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];

            int resized_w = GetResizedW(img);
            // 缩放图像到 (resized_w, img_h)
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h), interpolation: InterpolationFlags.Linear);
            unsafe
            {
                fixed (float* dst = inputData)
                {
                    ConvertToNormImg(resized_w, idx, img_c, img_h, img_w, resized, dst);
                }
            }
        }
        private int GetResizedW(Mat img)
        {
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
            return resized_w;
        }

        public void ResizeNormImg(Mat img, Mat resized, FixedBuffer buffer)
        {
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];

            int resized_w = GetResizedW(img);

            // 缩放图像到 (resized_w, img_h)
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h), interpolation: InterpolationFlags.Linear);

            unsafe
            {
                ConvertToNormImg(resized_w, 0, img_c, img_h, img_w, resized, buffer.Pointer);
            }
        }

        private OrtValue ResizeNormImgParallel(ImageIndex[] items, float[] batchData, long[] inputShape)
        {
            Parallel.For(0, items.Length, index =>
            {
                using Mat resized = new Mat();
                ResizeNormImg(items[index].Image, index, resized, batchData);
            });

            return OrtValue.CreateTensorValueFromMemory(batchData, inputShape);
        }

        public void PreprocessBatchParallelAsync(DisposableList<ImageIndex> imgCropList, int inputShapeSize, ChannelWriter<ClsPreResultBatchParallel> writer)
        {
            var batchList = imgCropList.Chunk(_ocrConfig.ClassifierConfig.ClsBatchNum).ToArray();
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
                int len = items.Length * inputShapeSize;
                float[] batchData = ArrayPool<float>.Shared.Rent(len);
                var ortVal = ResizeNormImgParallel(items, batchData, [items.LongLength, _clsImageShape[0], _clsImageShape[1], _clsImageShape[2]]);
                ClsPreResultBatchParallel res = new ClsPreResultBatchParallel(batchData, ortVal, dict[index]);
                writer.TryWrite(res);

            });

            writer.Complete();
        }




        public async Task PreprocessBatchAsync(DisposableList<ImageIndex> imgCropList, MatBufferPool matBuffer, DeviceType deviceType, ChannelWriter<ClsPreResultBatch> writer)
        {
            await PreprocessBatchBaseAsync(imgCropList, deviceType, matBuffer, writer, PreprocessChannel);
        }
        protected ClsPreResultBatch PreprocessChannel(ImageIndex batchImage, MatBufferPool matBuffer)
        {
            Mat img = batchImage.Image;
            var data = matBuffer.Rent();

            ResizeNormImg(img, data.ResizedImg, data.FixedBuffer);
            return new ClsPreResultBatch(data, batchImage);
        }


    }
}
