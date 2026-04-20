using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public abstract class TextDetectorBase
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        protected IDetPreprocess _detPreprocess;
        protected IDetPostprocess _detPostprocess;

        protected abstract IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue);

        public TextDetectorBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            _runOptions = new RunOptions();
            _session = session;
            _options = options;
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
