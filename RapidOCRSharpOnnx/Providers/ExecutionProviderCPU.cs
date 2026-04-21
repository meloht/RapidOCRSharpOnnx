using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public class ExecutionProviderCPU : ExecutionProvider
    {
        public ExecutionProviderCPU(OcrConfig ocrConfig) : base(ocrConfig)
        {
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;
            return sessionOptions;

        }

        protected override IOcrClassifier CreateOcrClassifier(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess)
        {
            return new TextClassifierOrtVal(session, options, postprocess, preprocess, OcrConfig);
        }

        protected override IOcrDetector CreateOcrDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            return new TextDetectorOrtVal(session, options, postprocess, preprocess);
        }

        protected override IOcrRecognizer CreateOcrRecognizer(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess)
        {
            return new TextRecognizerOrtVal(session, options, postprocess, preprocess, OcrConfig);
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }
    }
}
