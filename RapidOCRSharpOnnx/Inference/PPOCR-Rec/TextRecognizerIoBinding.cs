using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class TextRecognizerIoBinding : TextRecognizerBase, IOcrRecognizer
    {
        private OrtIoBinding _binding;
        public TextRecognizerIoBinding(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess, OcrConfig ocrConfig)
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
            return InferenceRunCore(inputOrtValue, _binding, perf);
        }
    }
}
