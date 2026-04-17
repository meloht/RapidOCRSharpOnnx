using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;


namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class TextDetector : IOcrDetector
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        private IDetPreprocess _detPreprocess;
        private IDetPostprocess _detPostprocess;


        public TextDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            _runOptions = new RunOptions();
            _session = session;
            _options = options;
            _detPreprocess = preprocess;
            _detPostprocess = postprocess;
        }
        public DetectResult Run(Mat image)
        {
            var data = _detPreprocess.Preprocess(image);
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data.Data, data.Dimensions);
            using var runOptions = new RunOptions();

            using var results = _session.Run(runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
            using var output0 = results[0];
            var res = _detPostprocess.PostProcess(image, output0);
            
            res.RatioW = data.RatioW;
            res.RatioH = data.RatioH;
            res.PaddingLeft = data.PaddingLeft;
            res.PaddingTop = data.PaddingTop;

            return res;
        }

        public void Dispose()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }
    }
}
