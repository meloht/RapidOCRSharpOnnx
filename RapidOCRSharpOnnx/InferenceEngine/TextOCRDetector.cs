using Clipper2Lib;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Config;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public class TextOCRDetector : IDisposable
    {
        private OrtInferSession _session;
        private const int _minSize = 3;
        private const int _BOX_SORT_Y_THRESHOLD = 10;
        System.Diagnostics.Stopwatch _timer;
        public TextOCRDetector()
        {
            _timer = new System.Diagnostics.Stopwatch();
            _session = new OrtInferSession(DetConfig.ModelPath);
        }


        public TextDetOutput Run(Mat image)
        {
            _timer.Restart();

            var data = Preprocess(image);

            using var outData = _session.RunInference(data);
            var res = DBPostProcess(outData, image.Height, image.Width);

            var boxs = SortedBoxes(res.boxes);
            _timer.Stop();

            return new TextDetOutput(boxs, res.scores, (int)_timer.ElapsedMilliseconds);
        }

        private DataTensorDimensions Preprocess(Mat image)
        {
            int maxWh = Math.Max(image.Width, image.Height);
            int limitSideLen = DetConfig.LimitSideLen;
            if (DetConfig.LimitType == LimitType.Min)
            {
                limitSideLen = DetConfig.LimitSideLen;
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

            int hh = resizedImg.Height;
            int ww = resizedImg.Width;
           
            float[] inputData = NormalizeAndPermute(resizedImg);
            return new DataTensorDimensions(inputData, new long[] { 1, 3, resizedImg.Height, resizedImg.Width });
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
            if (DetConfig.LimitType == LimitType.Max)
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
                throw new ResizeImgError("Image scaling failed: resizeW <= 0 or resizeH <= 0");
            // 4. 执行缩放并处理异常
            try
            {
                // 调用OpenCV缩放）
                Cv2.Resize(img, resized, new Size(resizeW, resizeH));
            }
            catch (Exception ex)
            {
                // 包装异常并保留原始异常（对应Python的raise ResizeImgError from exc）
                throw new ResizeImgError("Image scaling failed", ex);
            }

        }
        /// <summary>
        ///  归一化并转换为 Tensor (HWC -> CHW)
        /// </summary>
        private float[] NormalizeAndPermute(Mat img)
        {
            int len = img.Width * img.Height * 3;
            //float[] data = ArrayPool<float>.Shared.Rent(len);
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
                        data[index++] = ((float)vec[c]* scale - DetConfig.Mean[c]) / DetConfig.Std[c];
                    }
                }
            }
            return data;

        }
        /// <summary>
        /// DB算法后处理：概率图→二值化→轮廓提取→文本框生成
        /// </summary>
        /// <param name="output"></param>
        /// <param name="oriHeight"></param>
        /// <param name="oriWidth"></param>
        /// <returns></returns>
        public (List<Point2f[]> boxes, List<float> scores) DBPostProcess(OrtValue output, int oriHeight, int oriWidth)
        {
            var shape = output.GetTensorTypeAndShape().Shape;
            //获取OrtValue维度[1, 1, H, W]
            var dataArray = output.GetTensorDataAsSpan<float>().ToArray();
          
            int h = (int)shape[2];
            int w = (int)shape[3];

            Mat matPred = new Mat(h, w, MatType.CV_32FC1);
            matPred.SetArray(dataArray.ToArray());

            Mat mask = new Mat();
            Cv2.Threshold(matPred, mask, DetConfig.Thresh, 255, ThresholdTypes.Binary);
            mask.ConvertTo(mask, MatType.CV_8UC1); // 转为8位单通道（FindContours要求）

            if (DetConfig.UseDilation)
            {
                Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(2, 2));
                Cv2.Dilate(mask, mask, kernel);
            }
            var (boxes, scores) = ExtractBoxesFromMask(matPred, mask, oriWidth, oriHeight);

            // 5. 过滤无效框
            var (filteredBoxes, filteredScores) = FilterBoxes(boxes, scores, oriHeight, oriWidth);

            return (filteredBoxes, filteredScores);
        }

        // 可根据需要调整阈值，或作为参数传入
        private const double BOX_SORT_Y_THRESHOLD = 0.1;

        /// <summary>
        /// 对点列表进行排序：先按 Y 坐标从上到下，然后按 X 坐标从左到右，
        /// 并使用 Y 阈值将点划分为不同的行。
        /// </summary>
        /// <param name="dtBoxes">待排序的点列表（每个点代表一个边界框的左上角）</param>
        /// <returns>排序后的点列表</returns>
        public List<Point2f[]> SortedBoxes(List<Point2f[]> dtBoxes)
        {
            if (dtBoxes == null || dtBoxes.Count == 0)
                return dtBoxes ?? new List<Point2f[]>();

            // 1. 按第一个点的 Y 坐标稳定排序（OrderBy 默认稳定）
            var sortedByY = dtBoxes.OrderBy(box => box[0].Y).ToList();
            int n = sortedByY.Count;

            // 2. 分配行 ID：第一个点行 ID 为 0，后续若与前一个点 Y 差 >= 阈值，则换行
            int[] lineIds = new int[n];
            lineIds[0] = 0;
            for (int i = 1; i < n; i++)
            {
                float dy = sortedByY[i][0].Y - sortedByY[i - 1][0].Y;
                lineIds[i] = lineIds[i - 1] + (dy >= _BOX_SORT_Y_THRESHOLD ? 1 : 0);
            }

            // 3. 按行 ID 升序，同一行内按 X 坐标升序排序
            var result = sortedByY
                .Select((box, idx) => new { Box = box, LineId = lineIds[idx] })
                .OrderBy(item => item.LineId)          // 先按行号
                .ThenBy(item => item.Box[0].X)         // 再按 X 坐标
                .Select(item => item.Box)
                .ToList();

            return result;
        }
        /// <summary>
        /// 从二值化图提取检测框
        /// </summary>
        private (List<Point2f[]>, List<float>) ExtractBoxesFromMask(Mat matPred, Mat mask, int destWidth, int destHeight)
        {
            int height = mask.Rows;
            int width = mask.Cols;

            // 查找轮廓（对应cv2.findContours）

            OpenCvSharp.Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(mask, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            // 限制最大轮廓数量
            int numContours = Math.Min(contours.Length, DetConfig.MaxCandidates);

            List<Point2f[]> boxesList = new List<Point2f[]>();
            List<float> scoresList = new List<float>();

            for (int index = 0; index < numContours; index++)
            {
                OpenCvSharp.Point[] contour = contours[index];

                // 获取最小外接矩形和最小边长
                var (miniBox, minSide) = GetMiniBoxes(contour);
                if (minSide < _minSize)
                    continue;

                // 计算分数

                float score = DetConfig.ScoreMode == "fast"
                    ? BoxScoreFast(matPred, miniBox)
                    : BoxScoreSlow(matPred, contour);

                if (score < DetConfig.BoxThresh)
                    continue;

                // 膨胀多边形
                var expandedBox = UnclipBox(miniBox);
                var (unclipBox, unclipSside) = GetMiniBoxes(expandedBox);
                if (unclipSside < _minSize + 2)
                    continue;

                NormalizeBox(unclipBox, width, destWidth, height, destWidth);


                boxesList.Add(unclipBox);
                scoresList.Add(score);
            }

            return (boxesList, scoresList);
        }
        private (List<Point2f[]> boxes, List<float> scores) FilterBoxes(List<Point2f[]> boxes, List<float> scores, int imgH, int imgW)
        {
            List<Point2f[]> filteredBoxes = new List<Point2f[]>();
            List<float> filteredScores = new List<float>();

            if (boxes.Count == 0) return (filteredBoxes, filteredScores);

            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];

                // 1. 顺时针排序点
                OrderPointsClockwise(box);

                // 2. 裁剪到图像范围内
                ClipBox(box, imgH, imgW);

                // 3. 过滤过小的框
                float width = (float)Math.Sqrt(Math.Pow(box[1].X - box[0].X, 2) + Math.Pow(box[1].Y - box[0].Y, 2));
                float height = (float)Math.Sqrt(Math.Pow(box[3].X - box[0].X, 2) + Math.Pow(box[3].Y - box[0].Y, 2));
                if (width <= 3 || height <= 3) continue;

                filteredBoxes.Add(box);
                filteredScores.Add(scores[i]);
            }

            return (filteredBoxes, filteredScores);
        }
        /// <summary>
        /// 顺时针排序框的4个点
        /// </summary>
        private void OrderPointsClockwise(Point2f[] pts)
        {
            // 按X坐标排序
            Array.Sort(pts, (a, b) => a.X.CompareTo(b.X));
            Point2f[] left = pts.Take(2).ToArray();
            Point2f[] right = pts.Skip(2).ToArray();

            // 左半部分按Y排序
            Array.Sort(left, (a, b) => a.Y.CompareTo(b.Y));
            Point2f tl = left[0];
            Point2f bl = left[1];

            // 右半部分按Y排序
            Array.Sort(right, (a, b) => a.Y.CompareTo(b.Y));
            Point2f tr = right[0];
            Point2f br = right[1];

            // 重新赋值
            pts[0] = tl;
            pts[1] = tr;
            pts[2] = br;
            pts[3] = bl;
        }

        /// <summary>
        /// 裁剪框坐标到图像范围内
        /// </summary>
        private void ClipBox(Point2f[] box, int imgH, int imgW)
        {
            for (int i = 0; i < box.Length; i++)
            {
                box[i].X = Math.Clamp(box[i].X, 0, imgW - 1);
                box[i].Y = Math.Clamp(box[i].Y, 0, imgH - 1);
            }
        }

        /// <summary>
        /// 坐标归一化
        /// </summary>
        private void NormalizeBox(Point2f[] box, int srcW, int destW, int srcH, int destH)
        {
            for (int i = 0; i < box.Length; i++)
            {
                box[i].X = (float)Math.Round(box[i].X / srcW * destW);
                box[i].Y = (float)Math.Round(box[i].Y / srcH * destH);
                box[i].X = Math.Clamp(box[i].X, 0, destW);
                box[i].Y = Math.Clamp(box[i].Y, 0, destH);
            }
        }
        /// <summary>
        /// 多边形膨胀（Unclip）
        /// </summary>
        private OpenCvSharp.Point[] UnclipBox(Point2f[] box)
        {
            // 1. 计算多边形的面积和周长
            double area = Cv2.ContourArea(box);
            double length = Cv2.ArcLength(box, true); // true 表示封闭多边形
            double distance = area * DetConfig.UnclipRatio / length; // 扩张距离（像素单位）

            // 2. 将 OpenCV 点转换为 Clipper 所需的 Path64（使用 long 类型）

            Path64 path = new Path64();
            foreach (var p in box)
            {
                path.Add(new Point64((long)p.X, (long)p.Y));
            }

            // 3. 创建 ClipperOffset 对象，设置偏移参数
            ClipperOffset offset = new ClipperOffset();
            // JoinType.Round 对应 pyclipper.JT_ROUND；EndType.Polygon 对应 ET_CLOSEDPOLYGON
            offset.AddPath(path, JoinType.Round, EndType.Polygon);

            // 4. 执行偏移（注意偏移距离也需要乘以 scale）
            Paths64 solution = new Paths64();
            offset.Execute(distance, solution);

            // 5. 处理结果：通常我们取第一个扩张后的多边形（原 Python 代码返回单个多边形）
            if (solution.Count == 0)
                return new OpenCvSharp.Point[0]; // 没有结果时返回空数组

            // 将结果中的点缩放回原始坐标并转换为 OpenCV Point
            var expandedPath = solution[0]; // 根据需求，可能需要选择面积最大的轮廓或所有轮廓
            OpenCvSharp.Point[] result = expandedPath.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray();

            return result;
        }

        private float BoxScoreFast(Mat matPred, Point2f[] box)
        {
            // 计算框的边界
            float minX = box.Min(p => p.X);
            float maxX = box.Max(p => p.X);
            float minY = box.Min(p => p.Y);
            float maxY = box.Max(p => p.Y);

            int xMin = Math.Clamp((int)Math.Floor(minX), 0, matPred.Width - 1);
            int xMax = Math.Clamp((int)Math.Ceiling(maxX), 0, matPred.Width - 1);
            int yMin = Math.Clamp((int)Math.Floor(minY), 0, matPred.Height - 1);
            int yMax = Math.Clamp((int)Math.Ceiling(maxY), 0, matPred.Height - 1);

            // 创建掩码
            Mat mask = Mat.Zeros(new OpenCvSharp.Size(xMax - xMin + 1, yMax - yMin + 1), MatType.CV_8UC1);
            Point2f[] boxShifted = box.Select(p => new Point2f(p.X - xMin, p.Y - yMin)).ToArray();
            OpenCvSharp.Point[][] pts = [boxShifted.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray()];

            // 填充多边形
            Cv2.FillPoly(mask, pts, Scalar.White);


            Rect roi = new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);
            Mat roiMat = new Mat(matPred, roi);
            Scalar meanValue = Cv2.Mean(roiMat, mask);
            return (float)meanValue.Val0;
        }
        private float BoxScoreSlow(Mat matPred, OpenCvSharp.Point[] contour)
        {
            // 计算轮廓边界
            float minX = contour.Min(p => p.X);
            float maxX = contour.Max(p => p.X);
            float minY = contour.Min(p => p.Y);
            float maxY = contour.Max(p => p.Y);

            int xMin = Math.Clamp((int)minX, 0, matPred.Width - 1);
            int xMax = Math.Clamp((int)maxX, 0, matPred.Width - 1);
            int yMin = Math.Clamp((int)minY, 0, matPred.Height - 1);
            int yMax = Math.Clamp((int)maxY, 0, matPred.Height - 1);

            // 创建掩码
            Mat mask = Mat.Zeros(new OpenCvSharp.Size(xMax - xMin + 1, yMax - yMin + 1), MatType.CV_8UC1);
            OpenCvSharp.Point[] contourShifted = contour.Select(p => new OpenCvSharp.Point(p.X - xMin, p.Y - yMin)).ToArray();
            OpenCvSharp.Point[][] pts = [contourShifted];

            // 填充多边形
            Cv2.FillPoly(mask, pts, Scalar.White);
            Rect roi = new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1);
            Mat roiMat = new Mat(matPred, roi);
            Scalar meanValue = Cv2.Mean(roiMat, mask);
            return (float)meanValue.Val0;
        }
        /// <summary>
        /// 获取最小外接矩形并排序点（对应Python get_mini_boxes）
        /// </summary>
        private (Point2f[], float) GetMiniBoxes(OpenCvSharp.Point[] contour)
        {
            // 最小外接矩形（对应cv2.minAreaRect）
            RotatedRect rect = Cv2.MinAreaRect(contour);
            Point2f[] points = Cv2.BoxPoints(rect);
            Array.Sort(points, (a, b) => a.X.CompareTo(b.X));


            // 排序点（保证顺时针）
            Point2f[] box = new Point2f[4];
            int idx1 = points[1].Y > points[0].Y ? 0 : 1;
            int idx4 = points[1].Y > points[0].Y ? 1 : 0;
            int idx2 = points[3].Y > points[2].Y ? 2 : 3;
            int idx3 = points[3].Y > points[2].Y ? 3 : 2;

            box[0] = points[idx1];
            box[1] = points[idx2];
            box[2] = points[idx3];
            box[3] = points[idx4];

            return (box, Math.Min(rect.Size.Width, rect.Size.Height));

        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
