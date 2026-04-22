using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public class TextClassifierOrtVal : TextClassifierBase, IOcrClassifier
    {

        public TextClassifierOrtVal(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OcrConfig ocrConfig)
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
