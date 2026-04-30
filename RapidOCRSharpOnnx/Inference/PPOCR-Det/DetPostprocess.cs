using Clipper2Lib;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class DetPostprocess : IDetPostprocess
    {
        private const int _minSize = 3;
        private const int _BOX_SORT_Y_THRESHOLD = 10;
        private DetectorConfig _detConfig;
        public DetPostprocess(DetectorConfig detConfig)
        {
            _detConfig = detConfig;
        }

        public DetResult PostProcess(Mat image, OrtValue output)
        {
            var res = DBPostProcess(output, image.Height, image.Width);

            SortedBoxes(res.DetItems);
            var imgCropList = new DisposableList<ImageIndex>();

            for (int i = 0; i < res.DetItems.Length; i++)
            {
                var imgCrop = GetRotateCropImage(image, res.DetItems[i].Box);
                imgCropList.Add(new ImageIndex(imgCrop, i));
            }

            res.ImgCropList = imgCropList;
            return res;
        }

        private DetResult DBPostProcess(OrtValue output, int oriHeight, int oriWidth)
        {
            List<Point2f[]> boxes = new List<Point2f[]>();
            List<float> scores = new List<float>();

            var shape = output.GetTensorTypeAndShape().Shape;
            //获取OrtValue维度[1, 1, H, W]
            var dataArray = output.GetTensorDataAsSpan<float>();

            int h = (int)shape[2];
            int w = (int)shape[3];

            using Mat matPred = new Mat(h, w, MatType.CV_32FC1);
            matPred.SetArray(dataArray.ToArray());

            using Mat mask = new Mat();
            Cv2.Threshold(matPred, mask, _detConfig.Thresh, 255, ThresholdTypes.Binary);


            mask.ConvertTo(mask, MatType.CV_8UC1); // 转为8位单通道（FindContours要求）
            if (_detConfig.UseDilation)
            {
                using Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(2, 2));

                Cv2.Dilate(mask, mask, kernel);
            }
            mask.ConvertTo(mask, MatType.CV_8UC1, 255);

            return ExtractBoxesFromMask(matPred, mask, oriWidth, oriHeight);
        }

        /// <summary>
        /// 对点列表进行排序：先按 Y 坐标从上到下，然后按 X 坐标从左到右，并使用 Y 阈值将点划分为不同的行。
        /// </summary>
        /// <param name="dtBoxes">待排序的点列表（每个点代表一个边界框的左上角）</param>
        private void SortedBoxes(DetBoxItem[] dtBoxes)
        {
            if (dtBoxes == null || dtBoxes.Length == 0)
                return;

            // 1. 按第一个点的 Y 坐标稳定排序
            Array.Sort(dtBoxes, (a, b) => a.Box[0].Y.CompareTo(b.Box[0].Y));

            // 2. 分配行 ID：第一个点行 ID 为 0，后续若与前一个点 Y 差 >= 阈值，则换行
            dtBoxes[0].LineId = 0;
            for (int i = 1; i < dtBoxes.Length; i++)
            {
                float dy = dtBoxes[i].Box[0].Y - dtBoxes[i - 1].Box[0].Y;
                dtBoxes[i].LineId = dtBoxes[i - 1].LineId + (dy >= _BOX_SORT_Y_THRESHOLD ? 1 : 0);

            }
            // 3. 按行 ID 升序，同一行内按 X 坐标升序排序
            Array.Sort(dtBoxes, (a, b) =>
            {
                int lineCompare = a.LineId.CompareTo(b.LineId);
                if (lineCompare != 0)
                    return lineCompare;
                return a.Box[0].X.CompareTo(b.Box[0].X);
            });

        }

        private Mat GetRotateCropImage(Mat img, Point2f[] points)
        {
            // 计算宽度
            float width1 = UtilsHelper.Distance(points[0], points[1]);
            float width2 = UtilsHelper.Distance(points[2], points[3]);
            float imgCropWidth = Math.Max(width1, width2);

            // 计算高度
            float height1 = UtilsHelper.Distance(points[0], points[3]);
            float height2 = UtilsHelper.Distance(points[1], points[2]);
            float imgCropHeight = Math.Max(height1, height2);

            // 目标矩形
            Point2f[] ptsStd = new Point2f[]
            {
            new Point2f(0, 0),
            new Point2f(imgCropWidth, 0),
            new Point2f(imgCropWidth, imgCropHeight),
            new Point2f(0, imgCropHeight)
            };

            // 透视变换矩阵
            using Mat M = Cv2.GetPerspectiveTransform(points, ptsStd);

            // 透视变换
            Mat dstImg = new Mat();
            Cv2.WarpPerspective(
                src: img,
                dst: dstImg,
                m: M,
                dsize: new Size(imgCropWidth, imgCropHeight),
                flags: InterpolationFlags.Cubic,
                borderMode: BorderTypes.Replicate
            );

            int dstHeight = dstImg.Rows;
            int dstWidth = dstImg.Cols;

            // 如果是“竖长图”，旋转90°
            if ((float)dstHeight / dstWidth >= 1.5f)
            {
                Cv2.Rotate(dstImg, dstImg, RotateFlags.Rotate90Clockwise);
            }

            return dstImg;
        }

        /// <summary>
        /// 裁剪框坐标到图像范围内
        /// </summary>
        private void ClipBox(Point2f[] box, int imgH, int imgW)
        {
            float maxX = imgW - 1;
            float maxY = imgH - 1;
            for (int i = 0; i < box.Length; i++)
            {
                box[i].X = Math.Clamp(box[i].X, 0, maxX);
                box[i].Y = Math.Clamp(box[i].Y, 0, maxY);
            }
        }

        /// <summary>
        /// 从二值化图提取检测框
        /// </summary>
        private DetResult ExtractBoxesFromMask(Mat matPred, Mat mask, int destWidth, int destHeight)
        {
            int height = mask.Rows;
            int width = mask.Cols;

            OpenCvSharp.Point[][] contours;
            HierarchyIndex[] hierarchy;

            Cv2.FindContours(mask, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            // 限制最大轮廓数量
            int numContours = Math.Min(contours.Length, _detConfig.MaxCandidates);
            List<DetBoxItem> detPostprocessItems = new List<DetBoxItem>();


            for (int index = 0; index < numContours; index++)
            {
                OpenCvSharp.Point[] contour = contours[index];

                // 获取最小外接矩形和最小边长
                var (miniBox, minSide) = GetMiniBoxes(contour);
                if (minSide < _minSize)
                    continue;

                // 计算分数

                float score = _detConfig.ScoreMode == ScoreMode.FAST
                    ? BoxScoreFast(matPred, miniBox)
                    : BoxScoreSlow(matPred, contour);

                if (score < _detConfig.BoxThresh)
                    continue;

                // 膨胀多边形
                var expandedBox = UnclipBox(miniBox);
                var (unclipBox, unclipSside) = GetMiniBoxes(expandedBox);
                if (unclipSside < _minSize + 2)
                    continue;

                NormalizeBox(unclipBox, width, destWidth, height, destHeight);

                // 过滤无效框
                if (FilterBoxes(unclipBox, destHeight, destWidth))
                {
                    detPostprocessItems.Add(new DetBoxItem(unclipBox, score, 0, null));
                }
            }

            return new DetResult(detPostprocessItems.ToArray());
        }



        private bool FilterBoxes(Point2f[] box, int imgH, int imgW)
        {
            // 1. 裁剪到图像范围内
            ClipBox(box, imgH, imgW);

            // 2. 过滤过小的框
            float width = (float)Math.Sqrt(Math.Pow(box[1].X - box[0].X, 2) + Math.Pow(box[1].Y - box[0].Y, 2));
            float height = (float)Math.Sqrt(Math.Pow(box[3].X - box[0].X, 2) + Math.Pow(box[3].Y - box[0].Y, 2));
            if (width <= 3 || height <= 3)
            {
                return false;
            }
            return true;
        }

        /// <summary>
        /// 多边形膨胀（Unclip）
        /// </summary>
        private OpenCvSharp.Point2f[] UnclipBox(Point2f[] box)
        {
            // 1. 计算多边形的面积和周长
            double area = Cv2.ContourArea(box);
            double length = Cv2.ArcLength(box, true); // true 表示封闭多边形
            double distance = area * _detConfig.UnclipRatio / length; // 扩张距离（像素单位）

            // 2. 将 OpenCV 点转换为 Clipper 所需的 Path64（使用 long 类型）
            PathD pathd = new PathD();
            foreach (var p in box)
            {
                pathd.Add(new PointD(p.X, p.Y));
            }
            PathsD solution = Clipper.InflatePaths([pathd], distance, JoinType.Round, EndType.Polygon);
            if (solution == null || solution.Count == 0)
                return Array.Empty<Point2f>();

            PathD largest = solution.OrderByDescending(p => Math.Abs(Clipper.Area(p))).First();

            // 5. 处理结果：通常我们取第一个扩张后的多边形
            if (solution.Count == 0)
                return new OpenCvSharp.Point2f[0]; // 没有结果时返回空数组

            OpenCvSharp.Point2f[] result = new Point2f[largest.Count];
            for (int i = 0; i < largest.Count; i++)
            {
                result[i] = new Point2f((float)largest[i].x, (float)largest[i].y);
            }
            return result;
        }

        /// <summary>
        /// 坐标归一化
        /// </summary>
        private void NormalizeBox(Point2f[] box, int width, int destWidth, int height, int destHeight)
        {
            for (int i = 0; i < box.Length; i++)
            {
                float x = box[i].X;
                float y = box[i].Y;

                float newX = (float)Math.Round(x / width * destWidth);
                float newY = (float)Math.Round(y / height * destHeight);

                // clip
                newX = Math.Clamp(newX, 0, destWidth);
                newY = Math.Clamp(newY, 0, destHeight);

                box[i].X = newX;
                box[i].Y = newY;
            }
            //return destPoints;
        }
        private float BoxScoreSlow(Mat matPred, OpenCvSharp.Point[] contour)
        {
            // 计算轮廓边界
            float minX = contour.Min(p => p.X);
            float maxX = contour.Max(p => p.X);
            float minY = contour.Min(p => p.Y);
            float maxY = contour.Max(p => p.Y);

            int xMin = Math.Clamp((int)Math.Round(minX, 0), 0, matPred.Width - 1);
            int xMax = Math.Clamp((int)Math.Round(maxX, 0), 0, matPred.Width - 1);
            int yMin = Math.Clamp((int)Math.Round(minY, 0), 0, matPred.Height - 1);
            int yMax = Math.Clamp((int)Math.Round(maxY, 0), 0, matPred.Height - 1);

            // 创建掩码
            using Mat mask = Mat.Zeros(new OpenCvSharp.Size(xMax - xMin + 1, yMax - yMin + 1), MatType.CV_8UC1);
            OpenCvSharp.Point[] contourShifted = contour.Select(p => new OpenCvSharp.Point(p.X - xMin, p.Y - yMin)).ToArray();
            OpenCvSharp.Point[][] pts = [contourShifted];

            // 填充多边形
            Cv2.FillPoly(mask, pts, Scalar.White);
            Rect roi = new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);
            using Mat roiMat = new Mat(matPred, roi);
            Scalar meanValue = Cv2.Mean(roiMat, mask);
            return (float)meanValue.Val0;
        }

        private float BoxScoreFast(Mat matPred, Point2f[] box)
        {
            // 计算框的边界
            float minX = box.Min(p => p.X);
            float maxX = box.Max(p => p.X);
            float minY = box.Min(p => p.Y);
            float maxY = box.Max(p => p.Y);

            int xMin = Math.Clamp((int)Math.Round(Math.Floor(minX), 0), 0, matPred.Width - 1);
            int xMax = Math.Clamp((int)Math.Round(Math.Ceiling(maxX), 0), 0, matPred.Width - 1);
            int yMin = Math.Clamp((int)Math.Round(Math.Floor(minY), 0), 0, matPred.Height - 1);
            int yMax = Math.Clamp((int)Math.Round(Math.Ceiling(maxY), 0), 0, matPred.Height - 1);

            // 创建掩码
            using Mat mask = Mat.Zeros(new OpenCvSharp.Size(xMax - xMin + 1, yMax - yMin + 1), MatType.CV_8UC1);
            Point[] boxShifted = box.Select(p => new Point((int)Math.Round(p.X - xMin, 0), (int)Math.Round(p.Y - yMin, 0))).ToArray();
            OpenCvSharp.Point[][] pts = [boxShifted];

            // 填充多边形
            Cv2.FillPoly(mask, pts, Scalar.White);

            Rect roi = new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);
            using Mat roiMat = new Mat(matPred, roi);
            Scalar meanValue = Cv2.Mean(roiMat, mask);
            return (float)meanValue.Val0;
        }

        /// <summary>
        /// 获取最小外接矩形并排序点
        /// </summary>
        private (Point2f[], float) GetMiniBoxes(OpenCvSharp.Point2f[] contour)
        {
            // 最小外接矩形
            RotatedRect rect = Cv2.MinAreaRect(contour);
            Point2f[] points = Cv2.BoxPoints(rect);
            Array.Sort(points, (a, b) => a.X.CompareTo(b.X));


            // 排序点（保证顺时针）
            var idx1 = points[1].Y > points[0].Y ? points[0] : points[1];
            var idx4 = points[1].Y > points[0].Y ? points[1] : points[0];
            var idx2 = points[3].Y > points[2].Y ? points[2] : points[3];
            var idx3 = points[3].Y > points[2].Y ? points[3] : points[2];

            points[0] = idx1;
            points[1] = idx2;
            points[2] = idx3;
            points[3] = idx4;

            return (points, Math.Min(rect.Size.Width, rect.Size.Height));

        }

        /// <summary>
        /// 获取最小外接矩形并排序点
        /// </summary>
        private (Point2f[], float) GetMiniBoxes(OpenCvSharp.Point[] contour)
        {
            // 最小外接矩形
            RotatedRect rect = Cv2.MinAreaRect(contour);
            Point2f[] points = Cv2.BoxPoints(rect);
            Array.Sort(points, (a, b) => a.X.CompareTo(b.X));


            // 排序点（保证顺时针）
            var idx1 = points[1].Y > points[0].Y ? points[0] : points[1];
            var idx4 = points[1].Y > points[0].Y ? points[1] : points[0];
            var idx2 = points[3].Y > points[2].Y ? points[2] : points[3];
            var idx3 = points[3].Y > points[2].Y ? points[3] : points[2];

            points[0] = idx1;
            points[1] = idx2;
            points[2] = idx3;
            points[3] = idx4;

            return (points, Math.Min(rect.Size.Width, rect.Size.Height));

        }
    }
}
