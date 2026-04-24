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
    }
}
