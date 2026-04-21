using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
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


        public DetectResult TextDetect(Mat image)
        {
            using Mat resizedImg = image.Clone();
            var data = _detPreprocess.Preprocess(image, resizedImg);
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data.Data, data.Dimensions);

            using var output0 = InferenceRun(inputOrtValue);
            using var ortValue = output0[0];
            var res = _detPostprocess.PostProcess(resizedImg, ortValue);

            res.RatioW = data.RatioW;
            res.RatioH = data.RatioH;
            res.PaddingLeft = data.PaddingLeft;
            res.PaddingTop = data.PaddingTop;

            return res;
        }


        public void DisposeBase()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }
    }
}
