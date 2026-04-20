using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
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


        protected override IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue)
        {
            _binding.BindInput(_session.InputNames[0], inputOrtValue);
            _binding.BindOutputToDevice(_session.OutputNames[0], OrtMemoryInfo.DefaultInstance);
            _binding.SynchronizeBoundInputs();

            var results = _session.RunWithBoundResults(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            return results;
        }
    }
}
