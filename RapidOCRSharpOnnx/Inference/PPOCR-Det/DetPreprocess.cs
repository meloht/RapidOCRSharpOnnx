using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetPreprocess : IDetPreprocess
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
            RatioData ratio = ResizeImageWithinBounds(image, resizedImg, _ocrConfig.MinSideLen, _ocrConfig.MaxSideLen);
            PaddingData padding = ApplyVerticalPadding(resizedImg, _ocrConfig.WidthHeightRatio, _ocrConfig.MinHeight);

            var data = PreprocessImage(resizedImg);
            data.RatioW = ratio.RatioW;
            data.RatioH = ratio.RatioH;
            data.PaddingLeft = padding.Left;
            data.PaddingTop = padding.Top;

            return data;
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

            float[] inputData = NormalizeAndPermute(resizedImg);

            return new DetPreprocessData(inputData, [1, 3, resizedImg.Height, resizedImg.Width]);
        }



        /// <summary>
        ///  归一化并转换为 Tensor (HWC -> CHW)
        /// </summary>
        private float[] NormalizeAndPermute(Mat img)
        {
            int len = img.Width * img.Height * 3;
           
            float[] data = new float[len];
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
            try
            {
                // 调用OpenCV缩放）
                Cv2.Resize(img, resized, new Size(resizeW, resizeH));
            }
            catch (Exception ex)
            {
                // 包装异常并保留原始异常（对应Python的raise ResizeImgError from exc）
                throw new Exception("Image scaling failed", ex);
            }

        }

        /// <summary>
        /// 将图像尺寸限制在指定的最小和最大边长范围内，并确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像（不会被修改）</param>
        /// <param name="minSideLen">最小边长</param>
        /// <param name="maxSideLen">最大边长</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        private RatioData ResizeImageWithinBounds(Mat img, Mat resizedImg, float minSideLen, float maxSideLen)
        {
            int h = img.Height;
            int w = img.Width;

            RatioData ratio = new RatioData(1.0f, 1.0f);

            // 如果最大边超过上限，先缩小
            if (Math.Max(h, w) > maxSideLen)
            {
                ratio = ReduceMaxSide(img, resizedImg, maxSideLen);
               
                h = resizedImg.Height;
                w = resizedImg.Width;
            }

            // 如果最小边低于下限，再放大
            if (Math.Min(h, w) < minSideLen)
            {
                ratio = IncreaseMinSide(img, resizedImg, minSideLen);
            }

            return ratio;
        }

        /// <summary>
        /// 缩小图像使最大边不超过指定值，同时确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="maxSideLen">最大边长限制</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        private RatioData ReduceMaxSide(Mat img, Mat resizedImg, float maxSideLen = 2000)
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

            int resizeH = (int)(h * ratio);
            int resizeW = (int)(w * ratio);

            // 调整为32的倍数
            resizeH = (int)(Math.Round(resizeH / 32.0, MidpointRounding.AwayFromZero) * 32);
            resizeW = (int)(Math.Round(resizeW / 32.0, MidpointRounding.AwayFromZero) * 32);

            if (resizeH <= 0 || resizeW <= 0)
                throw new Exception("The adjusted width or height is less than or equal to 0");

            try
            {
                Cv2.Resize(img, resizedImg, new Size(resizeW, resizeH));
            }
            catch (Exception ex)
            {
                throw new Exception("Image scaling failed", ex);
            }

            float ratioH = h / (float)resizeH;
            float ratioW = w / (float)resizeW;

            return new RatioData(ratioH, ratioW);
        }

        /// <summary>
        /// 放大图像使最小边不低于指定值，同时确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="minSideLen">最小边长限制</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        private RatioData IncreaseMinSide(Mat img, Mat resizedImg, float minSideLen = 30)
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

            int resizeH = (int)(h * ratio);
            int resizeW = (int)(w * ratio);

            // 调整为32的倍数
            resizeH = (int)(Math.Round(resizeH / 32.0, MidpointRounding.AwayFromZero) * 32);
            resizeW = (int)(Math.Round(resizeW / 32.0, MidpointRounding.AwayFromZero) * 32);

            if (resizeH <= 0 || resizeW <= 0)
                throw new Exception("The adjusted width or height is less than or equal to 0");

            Cv2.Resize(img, resizedImg, new Size(resizeW, resizeH));

            float ratioH = h / (float)resizeH;
            float ratioW = w / (float)resizeW;

            return new RatioData(ratioH, ratioW);
        }


        private PaddingData ApplyVerticalPadding(Mat processedImg, float widthHeightRatio, float minHeight)
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
                return new PaddingData(paddingTop, paddingLeft);
            }
            else
            {
                // 返回原图像引用，表示未修改
                return new PaddingData(0, 0);
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
