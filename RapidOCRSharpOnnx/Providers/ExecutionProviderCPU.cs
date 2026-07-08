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
        public ExecutionProviderCPU(OcrConfig ocrConfig)
            : this(ocrConfig, null, null, null)
        {
        }
        public ExecutionProviderCPU(OcrConfig ocrConfig, SessionOptions detOpt = null, SessionOptions clsOpt = null, SessionOptions recOpt = null)
            : base(ocrConfig, detOpt, clsOpt, recOpt)
        {

        }

        protected override InferenceSession BuildInferenceSession(string modelPath, SessionOptions sessionOptions)
        {
            using SessionOptions opt = BuildSessionOptionsBase(sessionOptions);
            return new InferenceSession(modelPath, opt);
        }

        protected override IOcrClassifier CreateOcrClassifier(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess)
        {
            return new TextClassifierOrtVal(session, postprocess, preprocess, OcrConfig, GetDeviceType());
        }

        protected override IOcrDetector CreateOcrDetector(InferenceSession session, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            return new TextDetectorOrtVal(session, postprocess, preprocess, OcrConfig, GetDeviceType());
        }

        protected override IOcrRecognizer CreateOcrRecognizer(InferenceSession session, IRecPostprocess postprocess, IRecPreprocess preprocess)
        {
            return new TextRecognizerOrtVal(session, postprocess, preprocess, OcrConfig, GetDeviceType());
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }
    }
}
