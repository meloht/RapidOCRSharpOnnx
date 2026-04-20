using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class TextDetectorIoBinding : TextDetectorBase, IOcrDetector
    {
        private OrtIoBinding _binding;

        public TextDetectorIoBinding(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
            : base(session, options, postprocess, preprocess)
        {
            _binding = _session.CreateIoBinding();
        }

        public void Dispose()
        {
            DisposeBase();
            _binding?.Dispose();
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
