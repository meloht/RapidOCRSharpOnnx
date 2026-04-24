using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Utils
{
    public static class UtilsHelper
    {

        public static string GetFontPath(LangRec langRec)
        {
            string fontFileName = langRec switch
            {
                LangRec.CH => "FZYTK.TTF",
                LangRec.CH_DOC => "FZYTK.TTF",
                LangRec.EN => "FZYTK.TTF",
                LangRec.ARABIC => "arabic.ttf",
                LangRec.CHINESE_CHT => "chinese_cht.ttf",
                LangRec.CYRILLIC => "cyrillic.ttf",
                LangRec.ESLAV => "cyrillic.ttf",
                LangRec.DEVANAGARI => "devanagari_Martel-Regular.ttf",
                LangRec.JAPAN => "japan.ttc",
                LangRec.KOREAN => "korean.ttf",
                LangRec.KA => "kannada.ttf",
                LangRec.LATIN => "latin.ttf",
                LangRec.TA => "kannada.ttf",
                LangRec.TE => "telugu.ttf",
                LangRec.TH => "th.ttf",
                LangRec.EL => "el.ttf",
                _ => throw new ArgumentOutOfRangeException(nameof(langRec), $"Unsupported language: {langRec}")
            };

            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets", "fonts", fontFileName);
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

        public static float Distance(Point2f p1, Point2f p2)
        {
            float dx = p1.X - p2.X;
            float dy = p1.Y - p2.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }


        public static BoundedChannelOptions GetChannelOptions(int batchPoolSize)
        {
            var channelOptions = new BoundedChannelOptions(batchPoolSize)
            {
                SingleWriter = false,
                SingleReader = true,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.Wait
            };

            return channelOptions;
        }

        public static List<string> GetFilesFromDirectory(string path)
        {
            List<string> list = new List<string>();
            GetFiles(list, path);
            return list;

        }

        public static List<string> GetFilesFromListPaths(List<string> images)
        {
            List<string> list = new List<string>();
            foreach (var item in images)
            {
                string ext = Path.GetExtension(item);
                string fileExt = ext.ToLower();
                if (IsImageByExtension(fileExt))
                {
                    list.Add(item);
                }
            }
            return list;

        }

        public static void GetFiles(List<string> list, string path)
        {
            DirectoryInfo directory = new DirectoryInfo(path);
            var files = directory.GetFiles();

            foreach (var item in files)
            {
                string fileExt = item.Extension.ToLower();
                if (IsImageByExtension(fileExt))
                {
                    list.Add(item.FullName);
                }
            }
            var subDirectories = Directory.GetDirectories(path);

            foreach (string subDir in subDirectories)
            {
                GetFiles(list, subDir);
            }
        }
        public static bool IsImageByExtension(string ext)
        {
            return ext == ".jpg" || ext == ".jpeg" ||
                   ext == ".png" || ext == ".bmp" ||
                   ext == ".gif" || ext == ".tiff" ||
                   ext == ".webp";
        }

        public static string[] GetImageExtensions()
        {
            return new string[] { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp" };
        }

        public static SKBitmap MatToSKBitmapFast(Mat mat)
        {
            if (mat.Empty())
                throw new ArgumentException("Mat is empty");

            Mat converted = new Mat();

            if (mat.Channels() == 3)
            {
                Cv2.CvtColor(mat, converted, ColorConversionCodes.BGR2BGRA);
            }
            else if (mat.Channels() == 4)
            {
                converted = mat;
            }
            else if (mat.Channels() == 1)
            {
                Cv2.CvtColor(mat, converted, ColorConversionCodes.GRAY2BGRA);
            }
            else
            {
                throw new NotSupportedException("Unsupported format");
            }

            var bitmap = new SKBitmap(converted.Width, converted.Height, SKColorType.Bgra8888, SKAlphaType.Premul);

            unsafe
            {
                Buffer.MemoryCopy(
                    (void*)converted.DataPointer,
                    (void*)bitmap.GetPixels().ToPointer(),
                    converted.Total() * converted.ElemSize(),
                    converted.Total() * converted.ElemSize()
                );
            }

            return bitmap;
        }


        /// <summary>
        /// 计算文本框的高度（0号点与3号点的欧几里得距离）
        /// </summary>
        /// <param name="box">四边形坐标列表：[4个角点, x/y坐标]</param>
        /// <returns>文本框高度</returns>
        /// <exception cref="ArgumentException">坐标格式不合法时抛出异常</exception>
        public static float GetBoxHeight(Point2f[] boxes)
        {
            // 健壮性校验：确保坐标格式正确
            if (boxes == null || boxes.Length < 4)
            {
                throw new ArgumentException("文本框坐标格式无效，必须包含4个点，每个点2个坐标");
            }

            // 提取0号点和3号点的坐标
            float x0 = boxes[0].X;
            float y0 = boxes[0].Y;
            float x3 = boxes[3].X;
            float y3 = boxes[3].Y;

            // 计算欧几里得距离：√[(x0-x3)² + (y0-y3)²]
            float dx = x0 - x3;
            float dy = y0 - y3;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        public static float GetBoxWidth(Point2f[] boxes)
        {
            // 健壮性校验：确保坐标格式正确（和高度计算保持一致）
            if (boxes == null || boxes.Length < 4)
            {
                throw new ArgumentException("文本框坐标格式无效，必须包含4个点，每个点2个坐标");
            }

            // 提取0号点（左上）和1号点（右上）的坐标
            float x0 = boxes[0].X;
            float y0 = boxes[0].Y;
            float x1 = boxes[1].X;
            float y1 = boxes[1].Y;

            // 计算欧几里得距离：√[(x0-x1)² + (y0-y1)²]
            float dx = x0 - x1;
            float dy = y0 - y1;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }


    }
}
