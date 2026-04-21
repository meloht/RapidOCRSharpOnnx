using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Providers
{
    public class ExecutionProviderDirectML : ExecutionProvider
    {
        private int _deviceId;

        public ExecutionProviderDirectML(OcrConfig ocrConfig, int deviceId=0) : base(ocrConfig)
        {
            _deviceId = deviceId;
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML(this._deviceId);
            sessionOptions.EnableCpuMemArena = true;
            return sessionOptions;
        }

        protected override IOcrClassifier CreateOcrClassifier(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess)
        {
            return new TextClassifierIoBinding(session, options, postprocess, preprocess, OcrConfig);
        }

        protected override IOcrDetector CreateOcrDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            return new TextDetectorIoBinding(session, options, postprocess, preprocess);
        }

        protected override IOcrRecognizer CreateOcrRecognizer(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess)
        {
            return new TextRecognizerIoBinding(session, options, postprocess, preprocess, OcrConfig);
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }
    }
}
