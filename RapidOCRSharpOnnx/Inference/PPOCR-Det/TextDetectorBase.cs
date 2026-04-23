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
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;
using static System.Runtime.InteropServices.JavaScript.JSType;


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
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data.Data, data.Dimensions);
            _stopwatch.Stop();
            perf.Preprocess += _stopwatch.ElapsedMilliseconds;

            using var output0 = InferenceRun(inputOrtValue, perf);
            using var ortValue = output0[0];

            _stopwatch.Restart();
            var res = _detPostprocess.PostProcess(resizedImg, ortValue);

            res.ResizeData = data.ResizeData;

            ResultPerf<DetResult> result = new ResultPerf<DetResult>();
            result.Data = res;
            result.Perf = perf;
            _stopwatch.Stop();
            perf.Postprocess += _stopwatch.ElapsedMilliseconds;
            perf.SumTotal();
            return result;
        }


        public void BatchDetectAsync(List<string> listImg, ChannelWriter<OcrBatchResult> nextChannelWriter, OcrBatchResult[] batchResults)
        {
            Channel<DetPreResultBatch> channelDet = Channel.CreateBounded<DetPreResultBatch>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
            var producer = Task.Run(() => _detPreprocess.PreprocessBatchAsync(listImg, _deviceType, channelDet.Writer));

            var consumer = WriteNextAsync(channelDet, batchResults, nextChannelWriter);

            Task.WaitAll(producer, consumer);

            nextChannelWriter.Complete();

            Console.WriteLine("clsChannelWriter.Complete()");


        }
        private async Task WriteNextAsync(Channel<DetPreResultBatch> channelDet, OcrBatchResult[] batchResults, ChannelWriter<OcrBatchResult> nextChannelWriter)
        {
            int idx = 0;
            await foreach (DetPreResultBatch item in channelDet.Reader.ReadAllAsync())
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(item.PreResult.Data, item.PreResult.Dimensions);
                Console.WriteLine($"Detect batch {idx}");
                var output0 = InferenceRun(inputOrtValue, null);
                batchResults[idx] = new OcrBatchResult();
                // await BatchPostProcessAsync(output0, item, batchResults[idx], idx, nextChannelWriter);

                using (output0)
                using (item.ResizedImg)
                {
                    using var ortValue = output0[0];
                    var res = _detPostprocess.PostProcess(item.ResizedImg, ortValue);
                    res.ResizeData = item.PreResult.ResizeData;
                    batchResults[idx].DetResult = res;
                    Console.WriteLine($"Detect batch WriteAsync {idx} image count({res.ImgCropList.Count})");
                    await nextChannelWriter.WriteAsync(batchResults[idx]);
                }
                Interlocked.Increment(ref idx);

            }

        }

        private async Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, DetPreResultBatch item, OcrBatchResult batchResult, int idx, ChannelWriter<OcrBatchResult> writer)
        {
            await Task.Run(async () =>
             {
                 using (output)
                 using (item.ResizedImg)
                 {
                     using var ortValue = output[0];
                     var res = _detPostprocess.PostProcess(item.ResizedImg, ortValue);
                     res.ResizeData = item.PreResult.ResizeData;
                     batchResult.DetResult = res;
                     Console.WriteLine($"Detect batch WriteAsync {idx} image count({res.ImgCropList.Count})");
                     await writer.WriteAsync(batchResult);
                 }
             });

        }

    }
}
