using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public class TextClassifierIoBinding : TextClassifierBase, IOcrClassifier
    {
        private OrtIoBinding _binding;
        public TextClassifierIoBinding(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OcrConfig ocrConfig)
            : base(session, options, postprocess, preprocess, ocrConfig)
        {
            _binding = _session.CreateIoBinding();
        }

        public void Dispose()
        {
            DisposeBase();
            _binding.Dispose();
        }

        protected override IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue, PerfModel perf)
        {
            return InferenceRunCore(inputOrtValue,_binding, perf);
        }
    }
}
