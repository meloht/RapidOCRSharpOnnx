using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
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

            return PostProcess(resizedImg, ortValue, perf, data);
        }

        private ResultPerf<DetResult> PostProcess(Mat resizedImg,OrtValue ortValue, PerfModel perf, DetPreprocessData data)
        {
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



        protected async Task BatchDetectBaseAsync(List<string> listImg)
        {
       
            Channel<DetPreResultBatch> channelDet = Channel.CreateBounded<DetPreResultBatch>(GetChannelOptions(_ocrConfig.BatchPoolSize));
            //Channel<DetPreResultBatch> channelDet = Channel.CreateBounded<DetPreResultBatch>(GetChannelOptions(_ocrConfig.BatchPoolSize));

            var producer = _detPreprocess.PreprocessBatchAsync(listImg, _deviceType, channelDet.Writer);

            var consumer = Task.Run(async () =>
            {
                await foreach (DetPreResultBatch item in channelDet.Reader.ReadAllAsync())
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();

                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(item.PreResult.Data, item.PreResult.Dimensions);

                    var output0 = InferenceRun(inputOrtValue, null);

                    _ = BatchPostProcessAsync(output0, item);

                }
            });
            await Task.WhenAll(producer, consumer);

        }

        private async Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, DetPreResultBatch item)
        {
            await Task.Run(async () =>
            {
                using (output)
                using (item.resizedImg)
                {
                    using var ortValue = output[0];
                    var res = PostProcess(item.resizedImg,ortValue,new PerfModel(), item.PreResult);
                }
            });
        }

    }
}
