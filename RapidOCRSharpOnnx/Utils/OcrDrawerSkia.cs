using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Models;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace RapidOCRSharpOnnx.Utils
{
    internal class OcrDrawerSkia : IDisposable
    {
        private readonly SKTypeface _typeface;
        private readonly Random _rand = new Random(0);

        public float TextScore = 0.4f;
        private readonly TextCalRecBox _textCalRecBox;
        private readonly OcrConfig _ocrConfig;

        public OcrDrawerSkia(OcrConfig ocrConfig)
        {
            _ocrConfig = ocrConfig;
            string fontPath = UtilsHelper.GetFontPath(ocrConfig.RecognizerConfig.LangRec);
            _typeface = SKTypeface.FromFile(fontPath);
            _textCalRecBox = new TextCalRecBox(ocrConfig);
        }

        public void DrawTextBlock(Mat image, string savePath, DetResult detResult, RecResult[] recResults)
        {
            using var input = Convert(image);
            DrawTextBlockSKBitmap(input, savePath, detResult, recResults);
        }

        public void DrawTextBlock(string imagePath, string savePath, DetResult detResult, RecResult[] recResults)
        {
            using var image = SKBitmap.Decode(imagePath);
            DrawTextBlockSKBitmap(image, savePath, detResult, recResults);
        }
        private void DrawTextBlockSKBitmap(SKBitmap image, string savePath, DetResult detResult, RecResult[] recResults)
        {
            MapBoxesToOriginal(detResult, image.Height, image.Width);
            var croppedImgList = MapImgToOriginal(detResult.ImgCropList, detResult.ResizeData.RatioH, detResult.ResizeData.RatioW);
            var resCorp = _textCalRecBox.CalRecBoxes(croppedImgList, recResults, detResult.DetItems);

            SKBitmap result = null;
            if (resCorp == null || resCorp.Count == 0)
            {
                result = DrawOcrBoxTxt(image, detResult.DetItems);
            }
            else
            {
                result = DrawOcrBoxTxt(image, resCorp);
            }
            using var img = SKImage.FromBitmap(result);
            using var data = img.Encode(SKEncodedImageFormat.Jpeg, 100);
            if (File.Exists(savePath))
            {
                File.Delete(savePath);
            }
            File.WriteAllBytes(savePath, data.ToArray());
        }
        private Mat[] MapImgToOriginal(DisposableList<ImageIndex> imgs, float ratioH, float ratioW)
        {
            Mat[] results = new Mat[imgs.Count];
            for (int i = 0; i < imgs.Count; i++)
            {
                Mat img = imgs[i].Image;
                // 1. 获取当前图像的 高度、宽度
                int imgH = img.Rows;    // 图像高度
                int imgW = img.Cols;    // 图像宽度

                // 2. 计算原始图像尺寸
                int oriImgH = (int)Math.Round(imgH * ratioH, 0);
                int oriImgW = (int)Math.Round(imgW * ratioW, 0);

                // 3. 缩放回原始尺寸
                Mat resizeImg = new Mat();
                Cv2.Resize(img, resizeImg, new Size(oriImgW, oriImgH));

                results[i] = resizeImg;
            }
            return results;
        }
        private void MapBoxesToOriginal(DetResult det, int ori_h, int ori_w)
        {
            for (int i = 0; i < det.DetItems.Length; i++)
            {
                for (int j = 0; j < det.DetItems[i].Box.Length; j++)
                {
                    det.DetItems[i].Box[j].X = Math.Clamp((det.DetItems[i].Box[j].X - det.ResizeData.PaddingLeft) * det.ResizeData.RatioW, 0, ori_w);
                    det.DetItems[i].Box[j].Y = Math.Clamp((det.DetItems[i].Box[j].Y - det.ResizeData.PaddingTop) * det.ResizeData.RatioH, 0, ori_h);
                }
            }
        }
        private SKBitmap DrawOcrBoxTxt(SKBitmap image, IEnumerable<DetBoxItem> items)
        {
            int w = image.Width;
            int h = image.Height;

            // ===== 左图（copy）=====
            var imgLeft = new SKBitmap(w, h);
            using var canvasLeft = new SKCanvas(imgLeft);
            canvasLeft.DrawBitmap(image, 0, 0);

            // ===== 右图（白底）=====
            var imgRight = new SKBitmap(w, h);
            using var canvasRight = new SKCanvas(imgRight);
            canvasRight.Clear(SKColors.White);

            foreach (var item in items)
            {
                if (item.Score < TextScore)
                    continue;

                var box = item.Box;
                string txt = item.Word;

                var color = GetRandomColor();

                var points = box.Select(p => new SKPoint(p.X, p.Y)).ToArray();

                // ===== 左：填充 =====
                using (var paint = new SKPaint
                {
                    Color = color,
                    Style = SKPaintStyle.Fill,
                    IsAntialias = true
                })
                {
                    var path = BuildPath(points);
                    canvasLeft.DrawPath(path, paint);
                }

                // ===== 右：画框 =====
                using (var paint = new SKPaint
                {
                    Color = color,
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 1.0f,
                    IsAntialias = true
                })
                {
                    var path = BuildPath(points);
                    canvasRight.DrawPath(path, paint);
                }
                float boxH = UtilsHelper.GetBoxHeight(box);
                float boxW = UtilsHelper.GetBoxWidth(box);

                bool vertical = boxH > 2 * boxW;

                using SKFont font = GetFont(box, txt, boxH, boxW, vertical);
                // ===== 文本 =====
                DrawText(canvasRight, box, txt, font, boxH, boxW, vertical);
            }

            // ===== 半透明融合（左图）=====
            var blended = new SKBitmap(w, h);
            using (var canvas = new SKCanvas(blended))
            {
                canvas.DrawBitmap(image, 0, 0);

                using var paint = new SKPaint
                {
                    Color = new SKColor(255, 255, 255, 128) // 0.5 alpha
                };
                canvas.DrawBitmap(imgLeft, 0, 0, paint);
            }

            // ===== 拼接 =====
            var result = new SKBitmap(w * 2, h);
            using (var canvas = new SKCanvas(result))
            {
                canvas.DrawBitmap(blended, 0, 0);
                canvas.DrawBitmap(imgRight, w, 0);
            }

            return result;
        }

        private SKFont GetFont(Point2f[] box, string txt, float boxH, float boxW, bool vertical)
        {
            float fontSize = vertical
                ? Math.Max(boxW * 0.9f, 10)
                : Math.Max(boxH * 0.8f, 10);

            SKFont font = new(_typeface, fontSize);
            font.Size = FitTextSizeToHeight(font, boxH);
            return font;
        }

        // ===== 文本绘制 =====
        private void DrawText(SKCanvas canvas, Point2f[] box, string txt, SKFont font, float boxH, float boxW, bool vertical)
        {
            using var paint = new SKPaint
            {
                IsAntialias = true,
                Color = SKColors.Black
            };

            float x = box[3].X;
            float y = box[3].Y - font.Metrics.Descent;

            if (vertical)
            {
                float curY = y;
                foreach (char c in txt)
                {
                    string s = c.ToString();
                    canvas.DrawText(s, x + 3, curY, font, paint);
                    curY += GetTextBounds(font, c.ToString()).Height;
                }
            }
            else
            {
                canvas.DrawText(txt, x, y, font, paint);
            }
        }

        // ===== 构建多边形 =====
        private SKPath BuildPath(SKPoint[] pts)
        {
            var path = new SKPath();
            path.MoveTo(pts[0]);
            for (int i = 1; i < pts.Length; i++)
                path.LineTo(pts[i]);
            path.Close();
            return path;
        }

        private float GetBoxHeight(Point2f[] box)
        {
            float minY = box.Min(p => p.Y);
            float maxY = box.Max(p => p.Y);
            return maxY - minY;
        }
        private SKSize GetTextBounds(SKFont paint, string text)
        {
            if (paint == null || string.IsNullOrEmpty(text))
                return SKSize.Empty;

            // 测量文本边界矩形（自动计算宽高）
            paint.MeasureText(text, out SKRect rect);

            // 提取宽度、高度
            float width = rect.Width;
            float height = rect.Height;

            return new SKSize(width, height);
        }
        /// <summary>
        /// 计算检测框宽度
        /// </summary>
        private float GetBoxWidth(Point2f[] box)
        {
            float minX = box.Min(p => p.X);
            float maxX = box.Max(p => p.X);
            return maxX - minX;
        }
        private SKColor GetRandomColor()
        {
            return new SKColor(
                (byte)_rand.Next(0, 256),
                (byte)_rand.Next(0, 256),
                (byte)_rand.Next(0, 256)
            );
        }

        private float FitTextSizeToHeight(SKFont font, float targetHeight)
        {
            // 先设一个基准字体大小

            font.Size = 100;
            var metrics = font.Metrics;
            float textHeight = metrics.Descent - metrics.Ascent;

            // 计算缩放比例
            float scale = targetHeight / textHeight;

            return font.Size * scale;
        }


        private float FitTextSizeToWidth(SKFont font, string text, float targetWidth)
        {
            // 先设一个基准值
            font.Size = 100f;

            float measuredWidth = font.MeasureText(text);

            if (measuredWidth <= 0)
                return font.Size;

            float scale = targetWidth / measuredWidth;

            return font.Size * scale;
        }

        private float FitTextSizeBinary(SKFont font, string text, float targetWidth)
        {
            float low = 1f, high = 500f;

            while (high - low > 0.5f)
            {
                float mid = (low + high) / 2;
                font.Size = mid;

                float width = font.MeasureText(text);

                if (width > targetWidth)
                    high = mid;
                else
                    low = mid;
            }

            return low;
        }


        private float FitTextSizeBinary(SKFont font, float targetHeight)
        {
            float low = 1f, high = 500f;

            while (high - low > 0.5f)
            {
                float mid = (low + high) / 2;
                font.Size = mid;

                var m = font.Metrics;
                float h = m.Descent - m.Ascent;

                if (h > targetHeight)
                    high = mid;
                else
                    low = mid;
            }

            return low;
        }
        private SKBitmap MatToSKBitmapFast(Mat mat)
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

        private unsafe SKBitmap Convert(Mat mat)
        {
            if (mat.Empty())
                throw new ArgumentException("Mat empty");

            if (mat.Type() != MatType.CV_8UC3)
                throw new NotSupportedException("Only CV_8UC3 supported");

            int width = mat.Width;
            int height = mat.Height;

            var bitmap = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Premul);

            byte* src = (byte*)mat.DataPointer;
            byte* dst = (byte*)bitmap.GetPixels().ToPointer();

            int srcStride = (int)mat.Step();
            int dstStride = bitmap.RowBytes;

            if (Avx2.IsSupported)
            {
                ConvertAvx2(src, dst, width, height, srcStride, dstStride);
            }
            else
            {
                ConvertScalar(src, dst, width, height, srcStride, dstStride);
            }

            return bitmap;
        }

        private unsafe void ConvertAvx2(byte* src, byte* dst, int width, int height, int srcStride, int dstStride)
        {
            Vector256<byte> alpha = Vector256.Create((byte)255);

            for (int y = 0; y < height; y++)
            {
                byte* s = src + y * srcStride;
                byte* d = dst + y * dstStride;

                int x = 0;

                // 每次处理 8 像素（24字节 -> 32字节）
                for (; x <= width - 8; x += 8)
                {
                    // 加载 24 bytes（分三段）
                    Vector256<byte> v0 = Avx.LoadVector256(s);        // 实际会多读，但安全前提：padding足够
                    Vector256<byte> v1 = Avx.LoadVector256(s + 8);
                    Vector256<byte> v2 = Avx.LoadVector256(s + 16);

                    byte* tmp = (byte*)Marshal.AllocHGlobal(32);

                    for (int i = 0; i < 8; i++)
                    {
                        tmp[i * 4 + 0] = s[i * 3 + 0];
                        tmp[i * 4 + 1] = s[i * 3 + 1];
                        tmp[i * 4 + 2] = s[i * 3 + 2];
                        tmp[i * 4 + 3] = 255;
                    }

                    var result = Avx.LoadVector256(tmp);
                    Avx.Store(d, result);

                    s += 24;
                    d += 32;
                }

                // 尾部处理
                for (; x < width; x++)
                {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 255;

                    s += 3;
                    d += 4;
                }
            }
        }
        private unsafe void ConvertScalar(byte* src, byte* dst, int width, int height, int srcStride, int dstStride)
        {
            for (int y = 0; y < height; y++)
            {
                byte* s = src + y * srcStride;
                byte* d = dst + y * dstStride;

                for (int x = 0; x < width; x++)
                {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 255;

                    s += 3;
                    d += 4;
                }
            }
        }

        public void Dispose()
        {
            _typeface?.Dispose();
        }
    }
}
