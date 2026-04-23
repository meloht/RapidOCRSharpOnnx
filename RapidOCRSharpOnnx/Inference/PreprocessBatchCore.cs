using RapidOCRSharpOnnx.Providers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference
{
    public class PreprocessBatchCore<T1, T2, TResult>
    {
        protected void PreprocessBatchBaseAsync(List<T1> listImg, DeviceType deviceType, ChannelWriter<TResult> writer, T2 t2, Func<T1, T2, TResult> preprocess)
        {
            if (listImg == null || listImg.Count == 0)
            {
                writer.Complete();
                return;
            }
            var arr = GetPreprocessWorkersSize(listImg, deviceType);
            Task[] tasks = new Task[arr.Count()];
            int idx = 0;
            foreach (T1[] subList in arr)
            {
                tasks[idx++] = RunPreprocessSplitAsync(subList, writer, preprocess, t2);
            }

            Task.WaitAll(tasks);

            writer.Complete();
        }
        private Task RunPreprocessSplitAsync(IEnumerable<T1> list, ChannelWriter<TResult> writer, Func<T1, T2, TResult> preprocess, T2 t2)
        {
            return Task.Run(async () =>
            {
                foreach (T1 item in list)
                {
                    TResult res = preprocess(item, t2);
                    await writer.WriteAsync(res);
                }

            });
        }
        private IEnumerable<T1[]> GetPreprocessWorkersSize(List<T1> listImg, DeviceType deviceType)
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
                return [[.. listImg]];
            }
            return listImg.Chunk(size);
        }
    }
}
