using Clipper2Lib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualBasic;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Channels;



namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public abstract class TextDetectorBase : OnnxInferenceCore
    {
        protected IDetPreprocess _detPreprocess;
        protected IDetPostprocess _detPostprocess;

        public TextDetectorBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OcrConfig ocrConfig, DeviceType deviceType)
            : base(session, options, ocrConfig, deviceType)
        {
            _detPreprocess = preprocess;
            _detPostprocess = postprocess;
        }


        public ResultPerf<DetResult> TextDetect(Mat image)
        {
            PerfModel perf = new PerfModel();
            _stopwatch.Restart();
            using Mat resizedImg = image.Clone();
            var data = _detPreprocess.Preprocess(image, resizedImg);
            IDisposableReadOnlyCollection<OrtValue> output = null;
            try
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data.Data, data.Dimensions);
                _stopwatch.Stop();
                perf.Preprocess += _stopwatch.ElapsedMilliseconds;

                output = InferenceRun(inputOrtValue, perf);

                ArrayPool<float>.Shared.Return(data.Data, true);
                _stopwatch.Restart();
            }
            catch (Exception)
            {

                throw;
            }
            finally
            {
                if (data.Data != null)
                {
                    ArrayPool<float>.Shared.Return(data.Data, true);
                }
            }
            ResultPerf<DetResult> result = new ResultPerf<DetResult>();
            if (output != null)
            {
                using (output)
                {
                    using var ortValue = output[0];
                    var res = _detPostprocess.PostProcess(resizedImg, ortValue);

                    res.ResizeData = data.ResizeData;
                    result.Data = res;
                }
            }

            result.Perf = perf;
            _stopwatch.Stop();
            perf.Postprocess += _stopwatch.ElapsedMilliseconds;
            perf.SumTotal();
            return result;
        }


        public async Task BatchDetectAsync(List<string> listImg, ChannelWriter<OcrBatchResult> nextChannelWriter, OcrBatchResult[] batchResults)
        {
            List<ImagePathIndex> listPath = new List<ImagePathIndex>();
            for (int i = 0; i < listImg.Count; i++)
            {
                listPath.Add(new ImagePathIndex(listImg[i], i));
            }
            Channel<DetPreResultBatch> channelDet = Channel.CreateBounded<DetPreResultBatch>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
            var producer = Task.Run(async () => await _detPreprocess.PreprocessBatchAsync(listPath, _deviceType, channelDet.Writer));

            var consumer = WriteNextAsync(channelDet, batchResults, nextChannelWriter);

            await Task.WhenAll(producer, consumer);


        }
        private async Task WriteNextAsync(Channel<DetPreResultBatch> channelDet, OcrBatchResult[] batchResults, ChannelWriter<OcrBatchResult> nextChannelWriter)
        {
            ConcurrentBag<Task> tasks = new ConcurrentBag<Task>();
            await foreach (DetPreResultBatch item in channelDet.Reader.ReadAllAsync())
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(item.PreResult.Data, item.PreResult.Dimensions);

                long start = Stopwatch.GetTimestamp();

                var output0 = InferenceRun(inputOrtValue);
                long end = Stopwatch.GetTimestamp();
                batchResults[item.ImagePathIndex.Index].DetElapsedTime = (long)((end - start) * 1000.0 / Stopwatch.Frequency);

                var task = BatchPostProcessAsync(output0, item, batchResults[item.ImagePathIndex.Index], nextChannelWriter);
                tasks.Add(task);
            }

            await Task.WhenAll(tasks).ContinueWith(t => nextChannelWriter.Complete());
        }

        private Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, DetPreResultBatch item, OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> writer)
        {
            return Task.Run(async () =>
             {
                 ArrayPool<float>.Shared.Return(item.PreResult.Data, true);
                 using (output)
                 using (item.ResizedImg)
                 {
                     using var ortValue = output[0];
                     var res = _detPostprocess.PostProcess(item.ResizedImg, ortValue);
                     res.ResizeData = item.PreResult.ResizeData;
                     batchResult.DetResult = res;
                     Console.WriteLine($"Detect batch WriteAsync {item.ImagePathIndex.Index} image count({res.ImgCropList.Count})");
                     await writer.WriteAsync(batchResult);
                 }
             });

        }

    }
}
