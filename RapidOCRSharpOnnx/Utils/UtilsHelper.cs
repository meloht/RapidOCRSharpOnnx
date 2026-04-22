using OpenCvSharp;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

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

            var bitmap = new SKBitmap(
                converted.Width,
                converted.Height,
                SKColorType.Bgra8888,
                SKAlphaType.Premul
            );

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





    }
}
