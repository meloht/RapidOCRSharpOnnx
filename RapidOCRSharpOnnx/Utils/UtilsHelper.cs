using OpenCvSharp;
using RapidOCRSharpOnnx.Config;
using RapidOCRSharpOnnx.InferenceEngine;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RapidOCRSharpOnnx.Utils
{
    public static class UtilsHelper
    {
        public static void MakeDir(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
        }

        public static bool IsChineseChar(char ch)
        {
            // 对应Python的Unicode范围判断：
            // \u4e00-\u9fff：汉字
            // \u3000-\u303f：CJK标点（。、“”《》等）
            // \uff00-\uffef：全角符号（，．！？【】等）
            return (ch >= '\u4e00' && ch <= '\u9fff')
                   || (ch >= '\u3000' && ch <= '\u303f')
                   || (ch >= '\uff00' && ch <= '\uffef');
        }

        public static bool HasChineseChar(string text)
        {
            // 防护空值，避免NullReferenceException
            if (string.IsNullOrEmpty(text))
            {
                return false;
            }

            // LINQ的Any()等价于Python的any()：遍历每个字符，只要有一个满足就返回true
            return text.Any(ch => IsChineseChar(ch));
        }





        public static void SaveImg(string savePath, Mat img)
        {
            // 空值防护
            if (string.IsNullOrEmpty(savePath))
                throw new ArgumentNullException(nameof(savePath), "The save path cannot be empty");
            if (img == null || img.Empty())
                throw new ArgumentNullException(nameof(img), "The image data cannot be empty");

            // 1. 创建父目录（对应Python：Path(save_path).parent.mkdir(parents=True, exist_ok=True)）
            string parentDir = Path.GetDirectoryName(savePath);
            if (!string.IsNullOrEmpty(parentDir) && !Directory.Exists(parentDir))
            {
                Directory.CreateDirectory(parentDir); // 自动创建所有缺失的父目录，已存在不报错
            }

            // 2. 区分系统保存图片
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var extension = Path.GetExtension(savePath);
                var encoded = img.ImEncode(extension);
                File.WriteAllBytes(savePath, encoded);
            }
            else
            {
                // 非Windows系统：直接用ImWrite（对应Python的cv2.imwrite）
                Cv2.ImWrite(savePath, img);
            }
        }


        
        /// <summary>
        /// 将图像尺寸限制在指定的最小和最大边长范围内，并确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像（不会被修改）</param>
        /// <param name="minSideLen">最小边长</param>
        /// <param name="maxSideLen">最大边长</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        public static (Mat ResizedImg, double RatioH, double RatioW) ResizeImageWithinBounds(
            Mat img, double minSideLen, double maxSideLen)
        {
            int h = img.Height;
            int w = img.Width;
            double ratioH = 1.0, ratioW = 1.0;

            // 如果最大边超过上限，先缩小
            if (Math.Max(h, w) > maxSideLen)
            {
                var result = ReduceMaxSide(img, maxSideLen);
                img = result.ResizedImg;
                ratioH = result.RatioH;
                ratioW = result.RatioW;
                h = img.Height;
                w = img.Width;
            }

            // 如果最小边低于下限，再放大
            if (Math.Min(h, w) < minSideLen)
            {
                var result = IncreaseMinSide(img, minSideLen);
                img = result.ResizedImg;
                ratioH = result.RatioH;
                ratioW = result.RatioW;
            }

            return (img, ratioH, ratioW);
        }

        /// <summary>
        /// 缩小图像使最大边不超过指定值，同时确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="maxSideLen">最大边长限制</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        public static (Mat ResizedImg, double RatioH, double RatioW) ReduceMaxSide(
            Mat img, double maxSideLen = 2000)
        {
            int h = img.Height;
            int w = img.Width;

            double ratio = 1.0;
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
                throw new ResizeImgError("The adjusted width or height is less than or equal to 0");

            Mat resized = new Mat();
            try
            {
                Cv2.Resize(img, resized, new Size(resizeW, resizeH));
            }
            catch (Exception ex)
            {
                resized?.Dispose();
                throw new ResizeImgError("Image scaling failed", ex);
            }

            double ratioH = h / (double)resizeH;
            double ratioW = w / (double)resizeW;

            return (resized, ratioH, ratioW);
        }

        /// <summary>
        /// 放大图像使最小边不低于指定值，同时确保宽高为32的倍数
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="minSideLen">最小边长限制</param>
        /// <returns>调整后的图像及缩放比例（原始高/新高，原始宽/新宽）</returns>
        public static (Mat ResizedImg, double RatioH, double RatioW) IncreaseMinSide(
            Mat img, double minSideLen = 30)
        {
            int h = img.Height;
            int w = img.Width;

            double ratio = 1.0;
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
                throw new ResizeImgError("The adjusted width or height is less than or equal to 0");

            Mat resized = new Mat();
            try
            {
                Cv2.Resize(img, resized, new Size(resizeW, resizeH));
            }
            catch (Exception ex)
            {
                resized?.Dispose();
                throw new ResizeImgError("Image scaling failed", ex);
            }

            double ratioH = h / (double)resizeH;
            double ratioW = w / (double)resizeW;

            return (resized, ratioH, ratioW);
        }

        /// <summary>
        /// 添加圆角边框（实际是添加恒定值边框）
        /// </summary>
        /// <param name="img">输入图像</param>
        /// <param name="paddingTuple">四个方向的填充 (top, bottom, left, right)</param>
        /// <returns>添加边框后的图像</returns>
        public static Mat AddRoundLetterbox(Mat img, (int top, int bottom, int left, int right) paddingTuple)
        {
            // 使用常量值0（黑色）进行边框填充
            Mat paddedImg = new Mat();
            Cv2.CopyMakeBorder(
                img,
                paddedImg,
                paddingTuple.top,
                paddingTuple.bottom,
                paddingTuple.left,
                paddingTuple.right,
                BorderTypes.Constant,
                new Scalar(0, 0, 0) // 黑色填充，根据图像通道数自动适应
            );
            return paddedImg;
        }

        /// <summary>
        /// 计算垂直方向需要的填充高度
        /// </summary>
        /// <param name="h">原始高度</param>
        /// <param name="w">原始宽度</param>
        /// <param name="widthHeightRatio">宽高比</param>
        /// <param name="minHeight">最小高度</param>
        /// <returns>需要填充的高度（每侧填充量）</returns>
        public static int GetPaddingH(int h, int w, float widthHeightRatio, float minHeight)
        {
            int newH = (int)(Math.Max(w / widthHeightRatio, minHeight) * 2);
            int paddingH = (int)(Math.Abs(newH - h) / 2.0);
            return paddingH;
        }

        /// <summary>
        /// 根据宽高比和最小高度，决定是否在垂直方向添加填充
        /// </summary>
        /// <param name="img">输入图像（不会被修改）</param>
        /// <param name="opRecord">操作记录字典，用于记录填充信息</param>
        /// <param name="widthHeightRatio">宽高比阈值（如果为-1则忽略）</param>
        /// <param name="minHeight">最小高度阈值</param>
        /// <returns>处理后的图像和更新后的操作记录</returns>
        public static (Mat ProcessedImg, int paddingTop, int paddingLeft) ApplyVerticalPadding(
            Mat img,
            float widthHeightRatio,
            float minHeight)
        {

            int h = img.Height;
            int w = img.Width;
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
                Mat blockImg = AddRoundLetterbox(img, (paddingTop, paddingTop, 0, 0));
                paddingLeft = 0;
                return (blockImg, paddingTop, paddingLeft);
            }
            else
            {
                // 返回原图像引用，表示未修改
                return (img, 0, 0);
            }
        }
    }
}
