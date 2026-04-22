using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;


namespace RapidOCRSharpOnnx.Inference.PPOCR_Det
{
    public class TextDetectorOrtVal : TextDetectorBase, IOcrDetector
    {

        public TextDetectorOrtVal(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
            : base(session, options, postprocess, preprocess)
        {

        }

        public void Dispose()
        {
            DisposeBase();
        }

        protected override IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue, PerfModel perf)
        {
            return InferenceRunCore(inputOrtValue, perf);
        }
    }
}
