using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using RadpidOCRCSharpOnnx.Config;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public class TextClassifier
    {
        private readonly float[] _batchData;
        private readonly float[] imgData;
        private OrtInferSession _session;
        System.Diagnostics.Stopwatch _timer;
        public TextClassifier()
        {
            _batchData = new float[ClsConfig.ClsBatchNum * ClsConfig.ClsImageShape[0] * ClsConfig.ClsImageShape[1] * ClsConfig.ClsImageShape[2]];
            imgData = new float[ClsConfig.ClsImageShape[0] * ClsConfig.ClsImageShape[1] * ClsConfig.ClsImageShape[2]];

            _timer = new System.Diagnostics.Stopwatch();
            _session = new OrtInferSession(DetConfig.ModelPath);

        }

        public void TextClassify(Mat[] imgList)
        {
            int[] indices = new int[imgList.Length];
            float[] widthList = new float[imgList.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                indices[i] = i;
                widthList[i] = (float)imgList[i].Width / (float)imgList[i].Height;
            }

            Array.Sort(indices, (a, b) => widthList[a].CompareTo(widthList[b]));
            int imgCount = imgList.Length;
            ClsResult[] cls_res = new ClsResult[imgCount];
            for (int i = 0; i < imgCount; i++)
            {
                cls_res[i] = new ClsResult("", 0.0f);
            }

            int idx = 0;
            for (int i = 0; i < imgCount; i += ClsConfig.ClsBatchNum)
            {
                int endNo = Math.Min(imgCount, i + ClsConfig.ClsBatchNum);
                int batchSize = endNo - i;
                float[] batchData = _batchData;

                if (batchSize != ClsConfig.ClsBatchNum)
                {
                    batchData = new float[batchSize * ClsConfig.ClsImageShape[0] * ClsConfig.ClsImageShape[1] * ClsConfig.ClsImageShape[2]];
                }
                idx = 0;
                for (int j = i; j < endNo; j++)
                {
                    idx = ResizeNormImg(imgList[indices[j]], idx, batchData);
                }
                var input = new DataTensorDimensions(batchData, new long[] { batchSize, 3, ClsConfig.ClsImageShape[1], ClsConfig.ClsImageShape[2] });
                using var outData = _session.RunInference(input);
                var clsResults = ClsPostProcess(outData);
                for (int j = 0; j < clsResults.Length; j++)
                {
                    cls_res[indices[i + j]].Label = clsResults[j].Label;
                    cls_res[indices[i + j]].Score = clsResults[j].Score;
                    if (clsResults[j].Label == "180" && clsResults[j].Score > ClsConfig.ClsThresh)
                    {
                        Cv2.Rotate(imgList[indices[i + j]], imgList[indices[i + j]], RotateFlags.Rotate180);
                    }
                }

            }
        }

        public ClsResult[] ClsPostProcess(OrtValue ortValue)
        {

            var shapeInfo = ortValue.GetTensorTypeAndShape();
            int batchSize = (int)shapeInfo.Shape[0];
            int numClasses = (int)shapeInfo.Shape[1];

            var data = ortValue.GetTensorDataAsSpan<float>();
            if (data.Length != batchSize * numClasses)
                throw new InvalidOperationException("Data length mismatch.");

            ClsResult[] results = new ClsResult[batchSize];

            int rowStart = 0;
            int maxIdx = 0;
            for (int i = 0; i < batchSize; i++)
            {
                rowStart = i * numClasses;
                maxIdx = 0;
                float maxVal = data[rowStart];

                for (int j = 1; j < numClasses; j++)
                {
                    float val = data[rowStart + j];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
                results[i] = new ClsResult(ClsConfig.LabelList[maxIdx], maxVal);
            }
            return results;
        }

        public int ResizeNormImg(Mat img, int idx, float[] inputData)
        {
            // 获取原图尺寸和通道数
            int h = img.Height;
            int w = img.Width;
            int channels = img.Channels();
            int img_c = ClsConfig.ClsImageShape[0];
            int img_h = ClsConfig.ClsImageShape[1];
            int img_w = ClsConfig.ClsImageShape[2];

            // 计算缩放后的宽度（保持宽高比，但不超过目标宽度）
            float ratio = (float)w / h;
            double estimatedWidth = Math.Ceiling(img_h * ratio);

            int resized_w = estimatedWidth > img_w ? img_w : (int)Math.Ceiling(img_h * ratio);

            // 缩放图像到 (resized_w, img_h)
            Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h));

            // 转换为 float 类型并除以 255，得到 [0,1] 范围
            Mat resizedFloat = new Mat();
            if (channels == 1)
                resized.ConvertTo(resizedFloat, MatType.CV_32FC1, 1.0 / 255.0);
            else
                resized.ConvertTo(resizedFloat, MatType.CV_32FC3, 1.0 / 255.0);

            // 创建结果数组 (C, H, W)，初始值为 0（即填充部分自动为 0）
            float[,,] result = new float[img_c, img_h, img_w];

            Array.Fill(imgData, 0f); // 确保所有值初始化为 0
            // 将缩放后的图像数据复制到 result 中，并映射到 [-1, 1]
            if (img_c == 1)
            {
                // 灰度图：通道数 1
                for (int y = 0; y < img_h; y++)
                {
                    for (int x = 0; x < resized_w; x++)
                    {
                        // 读取像素值（已除以 255）
                        float val = resizedFloat.At<float>(y, x);
                        // 映射到 [-1, 1]： (val - 0.5) / 0.5 = val * 2 - 1
                        result[0, y, x] = val * 2f - 1f;

                    }
                    // 右侧超出 resized_w 的列保持 0
                }
            }
            else
            {
                // 彩色图：通道数 3（假设 OpenCV 默认 BGR 顺序）
                for (int y = 0; y < img_h; y++)
                {
                    for (int x = 0; x < resized_w; x++)
                    {
                        Vec3f pixel = resizedFloat.At<Vec3f>(y, x);
                        // 映射到 [-1, 1] 并按 (C, H, W) 顺序存储
                        result[0, y, x] = pixel.Item0 * 2f - 1f; // 通道 0 (B)
                        result[1, y, x] = pixel.Item1 * 2f - 1f; // 通道 1 (G)
                        result[2, y, x] = pixel.Item2 * 2f - 1f; // 通道 2 (R)


                    }
                }
            }


            for (int i = 0; i < img_c; i++)
            {
                for (int j = 0; j < img_h; j++)
                {
                    for (int k = 0; k < img_w; k++)
                    {
                        inputData[idx++] = result[i, j, k];
                    }
                }
            }
            return idx;
        }

        public float[] PreprocessNormImg(Mat img, int img_c, int img_h, int img_w)
        {
            // 获取原图尺寸和通道数
            int h = img.Height;
            int w = img.Width;
            int channels = img.Channels();

            // 计算缩放后的宽度（保持宽高比，但不超过目标宽度）
            float ratio = (float)w / h;
            double estimatedWidth = Math.Ceiling(img_h * ratio);

            int resized_w = estimatedWidth > img_w ? img_w : (int)Math.Ceiling(img_h * ratio);

            // 缩放图像到 (resized_w, img_h)
            Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h));

            using var canvas = new Mat(new OpenCvSharp.Size(img_w, img_h), MatType.CV_8UC3, new Scalar(0, 0, 0));
            resized.CopyTo(new Mat(canvas, new Rect(0, 0, resized_w, img_h)));
            // 2. 归一化并转换为 Tensor (HWC -> CHW)
            GetChwArr(canvas, imgData);
            return imgData;
        }

        private void GetChwArr(Mat paddedImg, float[] data)
        {
            int height = paddedImg.Height;
            int width = paddedImg.Width;
            int channels = paddedImg.Channels();
            int index = 0;
            for (int c = 0; c < channels; c++)          // 通道（R=0, G=1, B=2）
            {
                for (int h = 0; h < height; h++)  // 高度
                {
                    for (int w = 0; w < width; w++)  // 宽度
                    {
                        var vec = paddedImg.At<Vec3b>(h, w);
                        if (vec[c] != 0)
                        {
                            data[index++] = ((float)vec[c] / 255.0f) * 2f - 1f;
                        }
                        else
                        {
                            data[index++] = 0f;
                        }

                    }
                }
            }
        }
    }
}
