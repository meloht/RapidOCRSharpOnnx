using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class TextRecognizerOrtVal : TextRecognizerBase, IOcrRecognizer
    {

        public TextRecognizerOrtVal(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess, OcrConfig ocrConfig)
            : base(session, options, postprocess, preprocess, ocrConfig)
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
