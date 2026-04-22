using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public abstract class TextDetectorBase : OnnxInferenceCore
    {

        protected IDetPreprocess _detPreprocess;
        protected IDetPostprocess _detPostprocess;

        public TextDetectorBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
            : base(session, options)
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
            _stopwatch.Stop();
            perf.Postprocess += _stopwatch.ElapsedMilliseconds;

            res.RatioW = data.RatioW;
            res.RatioH = data.RatioH;
            res.PaddingLeft = data.PaddingLeft;
            res.PaddingTop = data.PaddingTop;

            perf.SumTotal();
            ResultPerf<DetResult> result = new ResultPerf<DetResult>();
            result.Data = res;
            result.Perf = perf;
            return result;
        }



    }
}
