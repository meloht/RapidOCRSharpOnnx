using OpenCvSharp;
using RapidOCRSharpOnnx.Providers;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference
{
    public class PreprocessBatchCore<T, TResult>
    {
        protected void PreprocessBatchBaseAsync(IList<T> listImg, DeviceType deviceType, ChannelWriter<TResult> writer, Func<T, TResult> preprocess)
        {
            if (listImg == null || listImg.Count == 0)
            {
                writer.Complete();
                return;
            }

            var arr = GetPreprocessWorkersSize(listImg, deviceType);
            Task[] tasks = new Task[arr.Count()];
            int idx = 0;
            foreach (T[] subList in arr)
            {
                tasks[idx++] = RunPreprocessSplitAsync(subList, writer, preprocess);
            }

            Task.WaitAll(tasks);

            writer.Complete();
        }
        private Task RunPreprocessSplitAsync(IList<T> list, ChannelWriter<TResult> writer, Func<T, TResult> preprocess)
        {
            return Task.Run(async () =>
            {
                foreach (T item in list)
                {
                    TResult res = preprocess(item);
                    await writer.WriteAsync(res);
                }

            });
        }
        private IEnumerable<T[]> GetPreprocessWorkersSize(IList<T> listImg, DeviceType deviceType)
        {
            int preprocessWorkers = Environment.ProcessorCount;
            if (deviceType == DeviceType.CPU)
            {
                preprocessWorkers = 2;
            }
            else
            {
                if (listImg.Count < Environment.ProcessorCount)
                {
                    preprocessWorkers = Environment.ProcessorCount / 2;
                }
                if (listImg.Count < preprocessWorkers)
                {
                    preprocessWorkers = 2;
                }
            }
            int size = listImg.Count / preprocessWorkers;

            if (size < 1)
            {
                size = listImg.Count;
            }
            if (size == 0)
            {
                return [listImg.ToArray()];
            }
            return listImg.Chunk(size);
        }

        protected unsafe void ConvertToNormImg(int resized_w, int index, int img_c, int img_h, int img_w, Mat resized, float[] inputData)
        {
            byte* src = (byte*)resized.DataPointer;
            int stride = (int)resized.Step();

            int hw = img_h * img_w;
            int chw = img_c * hw;
            int baseOffset = index * chw;

            float scale = 2.0f / 255.0f;

            int rOffset = baseOffset;
            int gOffset = baseOffset + hw;
            int bOffset = baseOffset + hw * 2;

            for (int y = 0; y < img_h; y++)
            {
                byte* row = src + y * stride;

                int x = 0;
                for (; x < resized_w; x++)
                {
                    int idx = y * img_w + x;
                    byte b = row[x * 3 + 0];
                    byte g = row[x * 3 + 1];
                    byte r = row[x * 3 + 2];

                    inputData[rOffset + idx] = r * scale - 1f;
                    inputData[gOffset + idx] = g * scale - 1f;
                    inputData[bOffset + idx] = b * scale - 1f;
                }
                for (; x < img_w; x++)
                {
                    int idx = y * img_w + x;

                    inputData[rOffset + idx] = 0f;
                    inputData[gOffset + idx] = 0f;
                    inputData[bOffset + idx] = 0f;
                }
            }

        }
    }
}
