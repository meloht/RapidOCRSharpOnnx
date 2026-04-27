using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;


namespace RapidOCRSharpOnnx.Inference
{
    public class TextCalRecBox
    {
        private OcrConfig _ocrConfig;
        public TextCalRecBox(OcrConfig ocrConfig)
        {
            _ocrConfig = ocrConfig;
        }
        /// <summary>
        /// 计算平均字符宽度
        /// </summary>
        /// <param name="wordCol">字符列索引列表</param>
        /// <param name="eachColWidth">每列的宽度</param>
        /// <returns>平均字符宽度，列表长度不足2时返回0</returns>
        private float CalcAvgCharWidth(int[] wordCol, float eachColWidth)
        {
            // 边界防护：列表元素数量小于2，避免除零错误（原Python代码会直接报错）
            if (wordCol == null || wordCol.Length < 2)
            {
                return 0.0f;
            }

            // 计算总长度：(最后一个索引 - 第一个索引) * 每列宽度
            float charTotalLength = (wordCol[^1] - wordCol[0]) * eachColWidth;

            // 计算平均值：总长度 / (元素数量 - 1)
            return charTotalLength / (wordCol.Length - 1);
        }

        /// <summary>
        /// 计算所有字符的平均宽度
        /// </summary>
        /// <param name="widthList">字符宽度列表</param>
        /// <param name="bboxX0">检测框左x坐标</param>
        /// <param name="bboxX1">检测框右x坐标</param>
        /// <param name="txtLen">文本长度</param>
        /// <returns>字符平均宽度</returns>
        private float CalcAllCharAvgWidth(float charWidthsAvg, int charCount, float bboxX0, float bboxX1, int txtLen)
        {
            // 1. 文本长度为0，直接返回0
            if (txtLen == 0)
            {
                return 0.0f;
            }

            // 2. 宽度列表非空，计算列表平均值
            if (charCount > 0)
            {
                return charWidthsAvg;
            }

            // 3. 无宽度列表，用检测框计算平均宽度
            return (bboxX1 - bboxX0) / txtLen;
        }



        /// <summary>
        /// 计算单个单词的字符单元格坐标
        /// </summary>
        /// <param name="lineCols">列索引列表</param>
        /// <param name="avgCharWidth">平均字符宽度</param>
        /// <param name="avgColWidth">平均列宽度</param>
        /// <param name="bboxPoints">外接矩形坐标 (x0,y0,x1,y1)</param>
        /// <returns>排序后的字符单元格三维坐标列表</returns>
        private Point2f[][] CalcBox(int[] lineCols, float avgCharWidth, float avgColWidth, Rect2fData bboxPoints)
        {
            Point2f[][] results = new Point2f[lineCols.Length][];

            // 遍历每个列索引
            for (int i = 0; i < lineCols.Length; i++)
            {
                results[i] = CalcBox(lineCols[i], avgCharWidth, avgColWidth, bboxPoints);
            }

            Array.Sort(results, (a, b) => a[0].X.CompareTo(b[0].X));

            // 按单元格左上角X坐标升序排序
            return results;
        }
        /// <summary>
        /// 计算单个单词的字符单元格坐标
        /// </summary>
        /// <param name="colIdx">列索引</param>
        /// <param name="avgCharWidth">平均字符宽度</param>
        /// <param name="avgColWidth">平均列宽度</param>
        /// <param name="bboxPoints">外接矩形坐标 (x0,y0,x1,y1)</param>
        /// <returns></returns>
        private Point2f[] CalcBox(int colIdx, float avgCharWidth, float avgColWidth, Rect2fData bboxPoints)
        {
            // 解包外接矩形坐标
            float x0 = bboxPoints.XMin;
            float y0 = bboxPoints.YMin;
            float x1 = bboxPoints.XMax;
            float y1 = bboxPoints.YMax;

            // 计算列中心点X坐标
            float centerX = (colIdx + 0.5f) * avgColWidth;
            float halfCharWidth = avgCharWidth / 2;

            // 计算字符左边界：取整后 ≥0，再叠加基准x0
            int tempX0 = (int)(centerX - halfCharWidth);
            float charX0 = Math.Max(tempX0, 0) + x0;

            // 计算字符右边界：取整后 ≤ 矩形宽度，再叠加基准x0
            int tempX1 = (int)(centerX + halfCharWidth);
            float maxWidth = x1 - x0;
            float charX1 = Math.Min(tempX1, maxWidth) + x0;

            // 构造四边形单元格：左上、右上、右下、左下
            Point2f[] cell = new Point2f[4];

            cell[0] = new Point2f(charX0, y0);
            cell[1] = new Point2f(charX1, y0);
            cell[2] = new Point2f(charX1, y1);
            cell[3] = new Point2f(charX0, y1);

            // 按单元格左上角X坐标升序排序
            return cell;
        }

        /// <summary>
        /// 将多个四边形坐标转换为最小外接矩形 (x_min, y_min, x_max, y_max)
        /// </summary>
        /// <param name="bbox">三维坐标列表：[N个四边形, 4个角点, x/y坐标]，对应shape (N,4,2)</param>
        /// <returns>最小外接矩形坐标元组</returns>
        /// <exception cref="ArgumentException">参数格式错误时抛出异常</exception>
        private Rect2fData QuadsToRectBbox(Point2f[][] bbox)
        {
            float xMin = float.MaxValue;
            float yMin = float.MaxValue;
            float xMax = float.MinValue;
            float yMax = float.MinValue;

            foreach (var item in bbox)
            {
                if (item.Length != 4)
                {
                    throw new ArgumentException("bbox shape must be (N, 4, 2)");
                }
                foreach (var point in item)
                {
                    xMin = Math.Min(xMin, point.X);
                    yMin = Math.Min(yMin, point.Y);
                    xMax = Math.Max(xMax, point.X);
                    yMax = Math.Max(yMax, point.Y);
                }
            }

            return new Rect2fData(xMin, yMin, xMax, yMax);
        }


        private Point2f[] CalcEnNumBox(int[] oneCol, float avgCharWidth, float avgColWidth, Rect2fData bboxPoints)
        {
            // 调用类内方法 calc_box
            Point2f[][] curWordCell = CalcBox(oneCol, avgCharWidth, avgColWidth, bboxPoints);

            // 调用四边形转矩形方法，获取最小外接矩形 (x0,y0,x1,y1)
            var rect = QuadsToRectBbox(curWordCell);

            // 严格按照原格式构造4个角点：左上、右上、右下、左下

            Point2f[] boxPoints = new Point2f[4];

            boxPoints[0] = new Point2f(rect.XMin, rect.YMin);
            boxPoints[1] = new Point2f(rect.XMax, rect.YMin);
            boxPoints[2] = new Point2f(rect.XMax, rect.YMax);
            boxPoints[3] = new Point2f(rect.XMin, rect.YMax);

            return boxPoints;
        }
        private List<WordItem> GetWordInfo(string text, List<int> validCol, List<float> confList)
        {
            if (!_ocrConfig.ReturnWordBox || validCol == null || validCol.Count == 0)
            {
                return null;
            }

            List<WordItem> wordItems = new List<WordItem>();

            float[] colWidth = new float[validCol.Count];
            for (int i = 1; i < validCol.Count; i++)
            {
                colWidth[i] = validCol[i] - validCol[i - 1];
            }

            int firstColValue = validCol[0];
            int minVal = text.Length > 0 && UtilsHelper.IsChineseChar(text[0]) ? 3 : 2;
            colWidth[0] = Math.Min(minVal, firstColValue);


            var wordContent = new List<char>();
            var wordColContent = new List<int>();
            var confArr = new List<float>();


            WordType? state = null;

            for (int cIdx = 0; cIdx < text.Length; cIdx++)
            {
                char ch = text[cIdx];

                // 处理空白字符：结束当前单词
                if (char.IsWhiteSpace(ch))
                {
                    if (wordContent.Count > 0)
                    {
                        WordItem wordItem = new WordItem
                        {
                            Words = wordContent.ToArray(),
                            WordCols = wordColContent.ToArray(),
                            WordType = state.Value,
                            Confs = confArr.ToArray()
                        };
                        wordItems.Add(wordItem);

                        wordContent.Clear();
                        wordColContent.Clear();
                        confArr.Clear();
                    }

                    continue;
                }


                // 判断当前字符类型
                WordType cState = UtilsHelper.IsChineseChar(ch) ? WordType.CN : WordType.EN_NUM;
                if (state == null)
                    state = cState;

                // 类型变化或列宽过大（>5）时切分单词
                if (state != cState || colWidth[cIdx] > 5)
                {
                    if (wordContent.Count > 0)
                    {

                        WordItem wordItem = new WordItem
                        {
                            Words = wordContent.ToArray(),
                            WordCols = wordColContent.ToArray(),
                            WordType = state.Value,
                            Confs = confArr.ToArray()
                        };
                        wordItems.Add(wordItem);

                        wordContent.Clear();
                        wordColContent.Clear();
                        confArr.Clear();

                    }
                    state = cState;
                }

                // 将当前字符加入正在构建的单词
                wordContent.Add(ch);
                wordColContent.Add(validCol[cIdx]);
                confArr.Add(confList[cIdx]);
            }

            // 处理最后一个单词
            if (wordContent.Count > 0)
            {
                WordItem wordItem = new WordItem
                {
                    Words = wordContent.ToArray(),
                    WordCols = wordColContent.ToArray(),
                    WordType = state.Value,
                    Confs = confArr.ToArray()
                };
                wordItems.Add(wordItem);
            }

            return wordItems;

        }

        public List<DetBoxItem> CalRecBoxes(Mat[] imgCropList, RecResult[] inferences, DetBoxItem[] items)
        {

            List<DetBoxItem> result = new List<DetBoxItem>();

            for (int i = 0; i < imgCropList.Length; i++)
            {
                string txt = inferences[i].Label;
                Mat img = imgCropList[i];

                int h = img.Height;
                int w = img.Width;
                Point2f[] imgBox = new Point2f[4];
                imgBox[0] = new Point2f(0, 0);
                imgBox[1] = new Point2f(w, 0);
                imgBox[2] = new Point2f(w, h);
                imgBox[3] = new Point2f(0, h);

                Point2f[] box = items[i].Box;
                var wordItems = GetWordInfo(txt, inferences[i].ValidCols, inferences[i].ConfList);

                var res = CalOcrWordBox(txt, imgBox, wordItems, inferences[i].LineTxtLen);
                AdjustBoxOverlap(res);
                var direction = GetBoxDirection(box);
                ReverseRotateCropImage(box.Select(p => new Point2f(p.X, p.Y)).ToArray(), res, direction);

                result.AddRange(res);

            }
            return result;
        }
        public void AdjustBoxOverlap(List<DetBoxItem> wordBoxList)
        {
            // 遍历到倒数第二个元素，防止索引越
            for (int i = 0; i < wordBoxList.Count - 1; i++)
            {
                Point2f[] cur = wordBoxList[i].Box;
                Point2f[] nxt = wordBoxList[i + 1].Box;

                // 判断条件：当前框右侧x坐标 > 下一个框左侧x坐标 → 存在重叠
                if (cur[1].X > nxt[0].X)
                {
                    // 计算重叠距离（绝对值）
                    float distance = Math.Abs(cur[1].X - nxt[0].X);

                    // cur[1][0] -= distance/2 （等价直接写distance/2）
                    cur[1].X -= distance / 2;
                    cur[2].X -= distance / 2;

                    // distance - distance/2 等价于 distance/2，简化后逻辑完全不变
                    nxt[0].X += distance / 2;
                    nxt[3].X += distance / 2;
                }
            }
        }

        private Direction GetBoxDirection(Point2f[] box)
        {
            // 校验输入：必须是4个点，每个点x/y两个坐标
            if (box.Length != 4)
                throw new ArgumentException("must be 4 point");

            // 计算四条边的长度（ 欧几里得距离）
            // 上边：box[0] → box[1]
            float topEdge = UtilsHelper.Distance(box[0], box[1]);
            // 右边：box[1] → box[2]
            float rightEdge = UtilsHelper.Distance(box[1], box[2]);
            // 下边：box[2] → box[3]
            float bottomEdge = UtilsHelper.Distance(box[2], box[3]);
            // 左边：box[3] → box[0]
            float leftEdge = UtilsHelper.Distance(box[3], box[0]);

            // 宽 = 上下边的最大值；高 = 左右边的最大值
            float width = MathF.Max(topEdge, bottomEdge);
            float height = MathF.Max(rightEdge, leftEdge);

            // 宽度极小值判断（避免除零）
            if (width < 1e-6f)
                return Direction.VERTICAL;

            // 计算宽高比，保留2位小数
            float aspectRatio = (float)Math.Round(height / width, 2);

            // 最终判断：≥1.5 垂直，否则水平
            return aspectRatio >= 1.5f ? Direction.VERTICAL : Direction.HORIZONTAL;
        }

        private void ReverseRotateCropImage(Point2f[] bboxPoints, List<DetBoxItem> wordBoxList, Direction direction)
        {
            // =====================计算最小左/上坐标，平移bbox =====================
            // 计算bbox所有点的最小X、最小Y
            float left = bboxPoints.Min(x => x.X);
            float top = bboxPoints.Min(x => x.Y);

            // 将bbox坐标平移到原点（减去left/top）
            Point2f[] srcPoints = new Point2f[4];
            for (int i = 0; i < 4; i++)
            {
                srcPoints[i] = new Point2f(bboxPoints[i].X - left, bboxPoints[i].Y - top);
            }

            // =====================计算裁剪图宽高 =====================
            float imgCropWidth = UtilsHelper.Distance(srcPoints[0], srcPoints[1]);
            float imgCropHeight = UtilsHelper.Distance(srcPoints[0], srcPoints[3]);

            // 标准四点坐标
            Point2f[] dstPoints = [new Point2f(0f, 0f), new Point2f(imgCropWidth, 0f), new Point2f(imgCropWidth, imgCropHeight), new Point2f(0, imgCropHeight)];

            using Mat perspectiveMat = Cv2.GetPerspectiveTransform(srcPoints, dstPoints);
            using Mat inverseMat = new Mat();
            Cv2.Invert(perspectiveMat, inverseMat);


            foreach (var wordPoints in wordBoxList)
            {
                Point2f[] newWordPoints = new Point2f[wordPoints.Box.Length];
                for (int j = 0; j < wordPoints.Box.Length; j++)
                {
                    float x = wordPoints.Box[j].X;
                    float y = wordPoints.Box[j].Y;
                    // 垂直方向：先旋转-90度 + X偏移
                    if (direction == Direction.VERTICAL)
                    {
                        // 旋转-90度（弧度）
                        var rotated = SRotate(MathF.PI / -2, x, y, 0, 0);
                        x = rotated[0] + imgCropWidth;
                        y = rotated[1];
                    }

                    // ===================== 齐次坐标 * 逆透视矩阵 =====================
                    // 构造齐次坐标 [x, y, 1]
                    float[] homoPoint = { x, y, 1 };
                    // 矩阵乘法：IM @ p
                    float[] result = MatMultiply(inverseMat, homoPoint);
                    float newX = result[0] / result[2]; // 归一化 x/z
                    float newY = result[1] / result[2]; // 归一化 y/z

                    // 平移回原图坐标 + 转int
                    int finalX = (int)(newX + left);
                    int finalY = (int)(newY + top);
                    newWordPoints[j] = new Point2f(finalX, finalY);
                }

                OrderPoints(newWordPoints);
                wordPoints.Box = newWordPoints;
            }

        }
        /// <summary>
        /// 绕指定点 (pointx, pointy) 顺时针旋转坐标
        /// 与 Python 原函数 s_rotate 逻辑完全一致
        /// </summary>
        /// <param name="angle">旋转角度（弧度）</param>
        /// <param name="valueX">待旋转点 X 坐标</param>
        /// <param name="valueY">待旋转点 Y 坐标</param>
        /// <param name="pointX">旋转中心点 X 坐标</param>
        /// <param name="pointY">旋转中心点 Y 坐标</param>
        /// <returns>旋转后的 [x, y] 坐标</returns>
        public float[] SRotate(float angle, float valueX, float valueY, float pointX, float pointY)
        {
            // 核心旋转公式（严格复刻Python原代码，顺时针旋转）
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);

            // 计算旋转后的X坐标
            float rotateX = (valueX - pointX) * cos + (valueY - pointY) * sin + pointX;
            // 计算旋转后的Y坐标
            float rotateY = (valueY - pointY) * cos - (valueX - pointX) * sin + pointY;

            // 返回结果（与Python [sRotatex, sRotatey] 格式一致）
            return new float[] { rotateX, rotateY };
        }

        /// <summary>
        /// 矩形框四点顺序排列
        /// </summary>
        /// <param name="oriBox">原始矩形框坐标 List[4个点][x,y]</param>
        /// <returns>排序后的标准矩形框坐标</returns>
        public void OrderPoints(Point2f[] oriBox)
        {
            // ===================== 2. 计算中心点 =====================
            double centerX = oriBox.Average(p => p.X);
            double centerY = oriBox.Average(p => p.Y);

            Point2f p1 = new Point2f(0, 0), p2 = new Point2f(0, 0), p3 = new Point2f(0, 0), p4 = new Point2f(0, 0);

            // ===================== 3. 分支1：有x=中心 且 有y=中心 → 菱形 =====================
            bool hasXEqualCenter = oriBox.Any(p => p.X == centerX);
            bool hasYEqualCenter = oriBox.Any(p => p.Y == centerY);
            if (hasXEqualCenter && hasYEqualCenter)
            {
                float minX = oriBox.Min(p => p.X);
                float minY = oriBox.Min(p => p.Y);
                float maxX = oriBox.Max(p => p.X);
                float maxY = oriBox.Max(p => p.Y);

                p1 = oriBox.First(p => p.X == minX);
                p2 = oriBox.First(p => p.Y == minY);
                p3 = oriBox.First(p => p.X == maxX);
                p4 = oriBox.First(p => p.Y == maxY);
            }
            // ===================== 分支2：所有x=中心 → 竖直线，按Y排序 =====================
            else if (oriBox.All(p => p.X == centerX))
            {
                // 按Y升序排列
                var sortedByY = oriBox.OrderBy(p => p.Y).ToArray();
                p1 = sortedByY[0];
                p2 = sortedByY[1];
                p3 = sortedByY[2];
                p4 = sortedByY[3];
            }
            // ===================== 分支3：有x=中心 且 所有y≠中心 → 先上下分，再左右 =====================
            else if (hasXEqualCenter && oriBox.All(p => p.Y != centerY))
            {
                // 按Y < 中心 / Y > 中心 分组
                var p12 = oriBox.Where(p => p.Y < centerY).ToArray();
                var p34 = oriBox.Where(p => p.Y > centerY).ToArray();

                float p12MinX = p12.Min(p => p.X);
                float p12MaxX = p12.Max(p => p.X);
                p1 = p12.First(p => p.X == p12MinX);
                p2 = p12.First(p => p.X == p12MaxX);

                float p34MaxX = p34.Max(p => p.X);
                float p34MinX = p34.Min(p => p.X);
                p3 = p34.First(p => p.X == p34MaxX);
                p4 = p34.First(p => p.X == p34MinX);
            }
            // ===================== 分支4：其他情况 → 先左右分，再上下 =====================
            else
            {
                // 按X < 中心 / X > 中心 分组
                var p14 = oriBox.Where(p => p.X < centerX).ToArray();
                var p23 = oriBox.Where(p => p.X > centerX).ToArray();

                float p14MinY = p14.Min(p => p.Y);
                float p14MaxY = p14.Max(p => p.Y);
                p1 = p14.First(p => p.Y == p14MinY);
                p4 = p14.First(p => p.Y == p14MaxY);

                float p23MinY = p23.Min(p => p.Y);
                float p23MaxY = p23.Max(p => p.Y);
                p2 = p23.First(p => p.Y == p23MinY);
                p3 = p23.First(p => p.Y == p23MaxY);
            }

            // ===================== 4. 转换输出=====================
            oriBox[0] = p1;
            oriBox[1] = p2;
            oriBox[2] = p3;
            oriBox[3] = p4;
        }

        /// <summary>
        /// 3x3矩阵 × 3维向量
        /// </summary>
        private static float[] MatMultiply(Mat mat, float[] vec)
        {
            double m00 = mat.At<double>(0, 0);
            double m01 = mat.At<double>(0, 1);
            double m02 = mat.At<double>(0, 2);
            double m10 = mat.At<double>(1, 0);
            double m11 = mat.At<double>(1, 1);
            double m12 = mat.At<double>(1, 2);
            double m20 = mat.At<double>(2, 0);
            double m21 = mat.At<double>(2, 1);
            double m22 = mat.At<double>(2, 2);

            // 矩阵向量乘法
            float x = (float)(m00 * vec[0] + m01 * vec[1] + m02 * vec[2]);
            float y = (float)(m10 * vec[0] + m11 * vec[1] + m12 * vec[2]);
            float z = (float)(m20 * vec[0] + m21 * vec[1] + m22 * vec[2]);
            return [x, y, z];
        }
        private float[] MatMultiply(double[,] mat, float[] vec)
        {
            float m00 = (float)mat[0, 0];
            float m01 = (float)mat[0, 1];
            float m02 = (float)mat[0, 2];
            float m10 = (float)mat[1, 0];
            float m11 = (float)mat[1, 1];
            float m12 = (float)mat[1, 2];
            float m20 = (float)mat[2, 0];
            float m21 = (float)mat[2, 1];
            float m22 = (float)mat[2, 2];

            // 矩阵向量乘法
            float x = m00 * vec[0] + m01 * vec[1] + m02 * vec[2];
            float y = m10 * vec[0] + m11 * vec[1] + m12 * vec[2];
            float z = m20 * vec[0] + m21 * vec[1] + m22 * vec[2];
            return [x, y, z];
        }
        private List<DetBoxItem> CalOcrWordBox(string recTxt, Point2f[] bbox, List<WordItem> wordItems, float lineTxtLen)
        {
            List<DetBoxItem> result = new List<DetBoxItem>();

            if (string.IsNullOrEmpty(recTxt) || wordItems == null || lineTxtLen == 0)
            {
                return result;
            }

            var bboxPoints = QuadsToRectBbox([bbox]);

            // 计算平均列宽度
            float avgColWidth = (bboxPoints.XMax - bboxPoints.XMin) / lineTxtLen;

            // 判断：是否全为英文/数字
            bool isAllEnNum = wordItems.All(v => v.WordType == WordType.EN_NUM);

            List<int[]> lineCols = new List<int[]>();
            List<int> lineCols2 = new List<int>();

            float charWidthsAvg = 0.0f;
            int charCount = 0;
            // 遍历单词 + 单词列索引
            for (int i = 0; i < wordItems.Count; i++)
            {
                var word = wordItems[i].Words;
                var wordCol = wordItems[i].WordCols;
                var conf = wordItems[i].Confs;
                // 全英文数字 + 不返回单字框 → 按单词处理
                if (isAllEnNum && !_ocrConfig.ReturnSingleCharBox)
                {
                    lineCols.Add(wordCol); // 

                    DetBoxItem item = new DetBoxItem(null, conf.Average(), 0, new string(word));
                    result.Add(item);
                }
                else
                {
                    // 汉字/中英混合 → 按单字符处理
                    lineCols2.AddRange(wordCol);

                    for (int j = 0; j < word.Length; j++)
                    {
                        DetBoxItem item = new DetBoxItem(null, conf[j], 0, word[j].ToString());
                        result.Add(item);
                    }
                }

                // 列长度为1，跳过平均宽度计算
                if (wordCol.Length == 1)
                    continue;

                // 计算当前单词的平均字符宽度
                float avgWidth = CalcAvgCharWidth(wordCol, avgColWidth);

                charWidthsAvg += avgWidth;
                charCount++;
            }
            charWidthsAvg = charWidthsAvg / charCount;

            // 计算全局平均字符宽度
            float avgCharWidth = CalcAllCharAvgWidth(charWidthsAvg, charCount, bboxPoints.XMin, bboxPoints.XMax, recTxt.Length);

            // 分支：英文单词框 / 单字符框
            if (isAllEnNum && !_ocrConfig.ReturnSingleCharBox)
            {
                for (int i = 0; i < lineCols.Count; i++)
                {
                    result[i].Box = CalcEnNumBox(lineCols[i], avgCharWidth, avgColWidth, bboxPoints);
                }

            }
            else
            {
                for (int i = 0; i < lineCols2.Count; i++)
                {
                    result[i].Box = CalcBox(lineCols2[i], avgCharWidth, avgColWidth, bboxPoints);
                }
                result.Sort((a, b) => a.Box[0].X.CompareTo(b.Box[0].X));

            }
            // 返回：文本内容、检测框、置信度
            return result;

        }
    }
}
