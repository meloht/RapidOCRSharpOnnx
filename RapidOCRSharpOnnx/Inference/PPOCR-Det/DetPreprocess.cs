using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetPreprocess : PreprocessBatchCore<ImagePathIndex, object, DetPreResultBatch>, IDetPreprocess
    {
        private OcrConfig _ocrConfig;
        private Scalar _paddingColor;
        public DetPreprocess(OcrConfig ocrConfig)
        {
            _ocrConfig = ocrConfig;
            _paddingColor = new Scalar(0, 0, 0);
        }
        public DetPreprocessData Preprocess(Mat image, Mat resizedImg)
        {
            ResizeData resizeData = new ResizeData();
            ResizeImageWithinBounds(image, resizedImg, _ocrConfig.MinSideLen, _ocrConfig.MaxSideLen, resizeData);
            ApplyVerticalPadding(resizedImg, _ocrConfig.WidthHeightRatio, _ocrConfig.MinHeight, resizeData);

            var data = PreprocessImage(resizedImg);
            data.ResizeData = resizeData;
            return data;
        }

        public async Task PreprocessBatchAsync(List<ImagePathIndex> listImg, DeviceType deviceType, ChannelWriter<DetPreResultBatch> writer)
        {
            await PreprocessBatchBaseAsync(listImg, deviceType, null, writer, PreprocessChannel);
        }
        protected DetPreResultBatch PreprocessChannel(ImagePathIndex imagePath, object t2)
        {
            using Mat img = Cv2.ImRead(imagePath.ImagePath);
            Mat resizedImg = img.Clone();
            var res = Preprocess(img, resizedImg);
            return new DetPreResultBatch(res, resizedImg, imagePath);
        }


        private DetPreprocessData PreprocessImage(Mat image)
        {
            int maxWh = Math.Max(image.Width, image.Height);
            int limitSideLen = _ocrConfig.DetectorConfig.LimitSideLen;
            if (_ocrConfig.DetectorConfig.LimitType == LimitType.Min)

            {
                limitSideLen = _ocrConfig.DetectorConfig.LimitSideLen;
            }
            else if (maxWh < 960)
            {
                limitSideLen = 960;
            }
            else if (maxWh < 1500)
            {
                limitSideLen = 1500;
            }
            else
            {
                limitSideLen = 2000;
            }

            using Mat resizedImg = new Mat();
            Resize(image, resizedImg, limitSideLen);
            //float[] inputData = new float[resizedImg.Height * resizedImg.Width * 3];
            int len = resizedImg.Height * resizedImg.Width * 3;
            float[] inputData = ArrayPool<float>.Shared.Rent(len);
            //NormalizeAndPermute(resizedImg, inputData);

            if (Avx2.IsSupported)
            {
                ToCHW_RGB_Normalized_AVX2(resizedImg, inputData);
            }
            else
            {
                ToCHW_RGB_Normalized(resizedImg, inputData);
            }

            return new DetPreprocessData(inputData, [1, 3, resizedImg.Height, resizedImg.Width]);
        }



        /// <summary>
        ///  归一化并转换为 Tensor (HWC -> CHW)
        /// </summary>
        private float[] NormalizeAndPermute(Mat img, float[] data)
        {
            int height = img.Height;
            int width = img.Width;
            int channels = img.Channels();
            float scale = 1.0f / 255.0f;
            int index = 0;
            for (int c = 0; c < channels; c++)          // 通道（R=0, G=1, B=2）
            {
                for (int h = 0; h < height; h++)  // 高度
                {
                    for (int w = 0; w < width; w++)  // 宽度
                    {
                        var vec = img.At<Vec3b>(h, w);
                        data[index++] = ((float)vec[c] * scale - _ocrConfig.DetectorConfig.Mean[c]) / _ocrConfig.DetectorConfig.Std[c];
                    }
                }
            }
            return data;

        }

        private unsafe void ToCHW_RGB_Normalized(Mat mat, float[] data)
        {
            int width = mat.Cols;
            int height = mat.Rows;
            int channels = mat.Channels();

            if (channels != 3)
                throw new ArgumentException("Only 3-channel images supported");

            byte* ptr = (byte*)mat.DataPointer;

            int hw = width * height;

            // 三个通道分开写（CHW）
            int rOffset = 0;
            int gOffset = hw;
            int bOffset = hw * 2;
            float scale = 1.0f / 255.0f;

            for (int y = 0; y < height; y++)
            {
                int rowOffset = y * width * channels;

                for (int x = 0; x < width; x++)
                {
                    int srcIndex = rowOffset + x * channels;

                    int dstIndex = y * width + x;

                    //  BGR -> RGB + 归一化 + CHW
                    data[rOffset + dstIndex] = ((float)ptr[srcIndex + 2] * scale - _ocrConfig.DetectorConfig.Mean[0]) / _ocrConfig.DetectorConfig.Std[0];
                    data[gOffset + dstIndex] = ((float)ptr[srcIndex + 1] * scale - _ocrConfig.DetectorConfig.Mean[1]) / _ocrConfig.DetectorConfig.Std[1];
                    data[bOffset + dstIndex] = ((float)ptr[srcIndex + 0] * scale - _ocrConfig.DetectorConfig.Mean[2]) / _ocrConfig.DetectorConfig.Std[2];
                }
            }
        }

        private unsafe void ToCHW_RGB_Normalized_AVX2(Mat mat, float[] data)
        {
            if (!Avx2.IsSupported)
                throw new NotSupportedException("AVX2 not supported");

            int width = mat.Width;
            int height = mat.Height;

            fixed (float* dst = data)
            {
                byte* src = (byte*)mat.DataPointer;

                int hw = width * height;

                float* dstR = dst;
                float* dstG = dst + hw;
                float* dstB = dst + hw * 2;
                float inv255 = 1.0f / 255.0f;

                Vector256<float> scale = Vector256.Create(1.0f / 255.0f);

                int stride = width * 3;
                int x = 0;
                for (int y = 0; y < height; y++)
                {
                    byte* row = src + y * stride;

                    x = 0;

                    // 每次处理 8 像素（24 字节）
                    for (; x <= width - 8; x += 8)
                    {
                        byte* p = row + x * 3;

                        // 手动加载（因为不是对齐的）
                        uint b0 = p[0]; uint g0 = p[1]; uint r0 = p[2];
                        uint b1 = p[3]; uint g1 = p[4]; uint r1 = p[5];
                        uint b2 = p[6]; uint g2 = p[7]; uint r2 = p[8];
                        uint b3 = p[9]; uint g3 = p[10]; uint r3 = p[11];
                        uint b4 = p[12]; uint g4 = p[13]; uint r4 = p[14];
                        uint b5 = p[15]; uint g5 = p[16]; uint r5 = p[17];
                        uint b6 = p[18]; uint g6 = p[19]; uint r6 = p[20];
                        uint b7 = p[21]; uint g7 = p[22]; uint r7 = p[23];

                        // 构建向量（R）
                        var vr = Vector256.Create(
                            (float)r0, (float)r1, (float)r2, (float)r3,
                            (float)r4, (float)r5, (float)r6, (float)r7);

                        var vg = Vector256.Create(
                            (float)g0, (float)g1, (float)g2, (float)g3,
                            (float)g4, (float)g5, (float)g6, (float)g7);

                        var vb = Vector256.Create(
                            (float)b0, (float)b1, (float)b2, (float)b3,
                            (float)b4, (float)b5, (float)b6, (float)b7);

                        // 归一化
                        vr = Avx.Multiply(vr, scale);
                        vg = Avx.Multiply(vg, scale);
                        vb = Avx.Multiply(vb, scale);

                        int idx = y * width + x;

                        Avx.Store(dstR + idx, vr);
                        Avx.Store(dstG + idx, vg);
                        Avx.Store(dstB + idx, vb);
                    }

                    // 处理尾部
                    for (; x < width; x++)
                    {
                        byte* p = row + x * 3;

                        int idx = y * width + x;

                        dstR[idx] = p[2] * inv255;
                        dstG[idx] = p[1] * inv255;
                        dstB[idx] = p[0] * inv255;
                    }
                }
            }
        }

        private void Resize(Mat img, Mat resized, int limitSideLen)
        {
            // 空值防护：输入图像为空/无效时抛出异常
            if (img == null || img.Empty())
                throw new ArgumentNullException(nameof(img), "The input image cannot be empty or invalid");

            // 1. 获取图像高和宽
            int h = img.Height;
            int w = img.Width;
            double ratio = 1.0;

            // 2. 根据LimitType计算缩放比例
            if (_ocrConfig.DetectorConfig.LimitType == LimitType.Max)
            {
                int maxSide = Math.Max(h, w);
                if (maxSide > limitSideLen)
                {
                    ratio = (double)limitSideLen / maxSide;
                }
                // 否则ratio保持1.0
            }
            else // LimitType.Min
            {
                int minSide = Math.Min(h, w);
                if (minSide < limitSideLen)
                {
                    ratio = (double)limitSideLen / minSide;
                }
                // 否则ratio保持1.0
            }

            // 3. 计算缩放后的高宽，并调整为32的整数倍（四舍五入后乘32）
            int resizeH = (int)(h * ratio);
            int resizeW = (int)(w * ratio);

            // 调整为32的整数倍：round(resize_h/32)*32
            resizeH = (int)Math.Round(resizeH / 32.0) * 32;
            resizeW = (int)Math.Round(resizeW / 32.0) * 32;

            // 边界检查：宽高<=0返回null
            if (resizeW <= 0 || resizeH <= 0)
                throw new Exception("Image scaling failed: resizeW <= 0 or resizeH <= 0");
            // 4. 执行缩放并处理异常
            Cv2.Resize(img, resized, new Size(resizeW, resizeH));

        }

        /// <summary>
        /// 将图像尺寸限制在指定的最小和最大边长范围内，并确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像（不会被修改）</param>
        /// <param name="minSideLen">最小边长</param>
        /// <param name="maxSideLen">最大边长</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        private void ResizeImageWithinBounds(Mat img, Mat resizedImg, float minSideLen, float maxSideLen, ResizeData resizeData)
        {
            int h = img.Height;
            int w = img.Width;

            resizeData.RatioW = 1.0f;
            resizeData.RatioH = 1.0f;

            // 如果最大边超过上限，先缩小
            if (Math.Max(h, w) > maxSideLen)
            {
                ReduceMaxSide(img, resizedImg, resizeData, maxSideLen);

                h = resizedImg.Height;
                w = resizedImg.Width;
            }

            // 如果最小边低于下限，再放大
            if (Math.Min(h, w) < minSideLen)
            {
                IncreaseMinSide(img, resizedImg, resizeData, minSideLen);
            }
        }

        /// <summary>
        /// 缩小图像使最大边不超过指定值，同时确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="maxSideLen">最大边长限制</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        private void ReduceMaxSide(Mat img, Mat resizedImg, ResizeData resizeData, float maxSideLen = 2000)
        {
            int h = img.Height;
            int w = img.Width;

            float ratio = 1.0f;
            if (Math.Max(h, w) > maxSideLen)
            {
                if (h > w)
                    ratio = maxSideLen / h;
                else
                    ratio = maxSideLen / w;
            }

            int resizeH = (int)Math.Round((h * ratio), 0, MidpointRounding.AwayFromZero);
            int resizeW = (int)Math.Round((w * ratio), 0, MidpointRounding.AwayFromZero);

            // 调整为32的倍数
            resizeH = (int)Math.Round(resizeH / 32.0, 0, MidpointRounding.AwayFromZero) * 32;
            resizeW = (int)Math.Round(resizeW / 32.0, 0, MidpointRounding.AwayFromZero) * 32;

            if (resizeH <= 0 || resizeW <= 0)
                throw new Exception("The adjusted width or height is less than or equal to 0");

            if (img.Width != resizeW || img.Height != resizeH)
            {
                Cv2.Resize(img, resizedImg, new Size(resizeW, resizeH));
            }

            float ratioH = h / (float)resizeH;
            float ratioW = w / (float)resizeW;
            resizeData.RatioW = ratioW;
            resizeData.RatioH = ratioH;

        }

        /// <summary>
        /// 放大图像使最小边不低于指定值，同时确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="minSideLen">最小边长限制</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        private void IncreaseMinSide(Mat img, Mat resizedImg, ResizeData resizeData, float minSideLen = 30)
        {
            int h = img.Height;
            int w = img.Width;

            float ratio = 1.0f;
            if (Math.Min(h, w) < minSideLen)
            {
                if (h < w)
                    ratio = minSideLen / h;
                else
                    ratio = minSideLen / w;
            }

            int resizeH = (int)Math.Round((h * ratio), 0, MidpointRounding.AwayFromZero);
            int resizeW = (int)Math.Round((w * ratio), 0, MidpointRounding.AwayFromZero);

            // 调整为32的倍数
            resizeH = (int)Math.Round(resizeH / 32.0, 0, MidpointRounding.AwayFromZero) * 32;
            resizeW = (int)Math.Round(resizeW / 32.0, 0, MidpointRounding.AwayFromZero) * 32;

            if (resizeH <= 0 || resizeW <= 0)
                throw new Exception("The adjusted width or height is less than or equal to 0");

            if (img.Width != resizeW || img.Height != resizeH)
            {
                Cv2.Resize(img, resizedImg, new Size(resizeW, resizeH));
            }

            float ratioH = h / (float)resizeH;
            float ratioW = w / (float)resizeW;

            resizeData.RatioH = ratioH;
            resizeData.RatioW = ratioW;
        }


        private void ApplyVerticalPadding(Mat processedImg, float widthHeightRatio, float minHeight, ResizeData resizeData)
        {
            int h = processedImg.Height;
            int w = processedImg.Width;
            int paddingTop = 0;
            int paddingLeft = 0;
            bool useLimitRatio;
            if (widthHeightRatio == -1)
                useLimitRatio = false;
            else
                useLimitRatio = w / (float)h > widthHeightRatio;

            if (h <= minHeight || useLimitRatio)
            {
                paddingTop = GetPaddingH(h, w, widthHeightRatio, minHeight);
                AddRoundLetterbox(processedImg, processedImg, paddingTop, paddingTop, 0, 0);

                paddingLeft = 0;
                resizeData.PaddingLeft = paddingLeft;
                resizeData.PaddingTop = paddingTop;

            }
            else
            {
                // 返回原图像引用，表示未修改
                resizeData.PaddingLeft = 0;
                resizeData.PaddingTop = 0;
            }
        }

        /// <summary>
        /// 计算垂直方向需要的填充高度
        /// </summary>
        /// <param name="h">原始高度</param>
        /// <param name="w">原始宽度</param>
        /// <param name="widthHeightRatio">宽高比</param>
        /// <param name="minHeight">最小高度</param>
        /// <returns>需要填充的高度（每侧填充量）</returns>
        private int GetPaddingH(int h, int w, float widthHeightRatio, float minHeight)
        {
            int newH = (int)(Math.Max(w / widthHeightRatio, minHeight) * 2);
            int paddingH = (int)(Math.Abs(newH - h) / 2.0);
            return paddingH;
        }

        /// <summary>
        /// 添加圆角边框（实际是添加恒定值边框）
        /// </summary>
        private void AddRoundLetterbox(Mat img, Mat processedImg, int top, int bottom, int left, int right)
        {
            // 使用常量值0（黑色）进行边框填充
            Cv2.CopyMakeBorder(
                img,
                processedImg,
                top,
                bottom,
                left,
                right,
                BorderTypes.Constant,
                _paddingColor // 黑色填充，根据图像通道数自动适应
            );

        }

    }
}
