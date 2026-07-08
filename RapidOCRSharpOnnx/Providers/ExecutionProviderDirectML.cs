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

        public ExecutionProviderDirectML(OcrConfig ocrConfig, int deviceId = 0)
            : this(ocrConfig, deviceId, null, null, null)
        {
        }

        public ExecutionProviderDirectML(OcrConfig ocrConfig, int deviceId = 0, SessionOptions detOpt = null, SessionOptions clsOpt = null, SessionOptions recOpt = null)
            : base(ocrConfig, detOpt, clsOpt, recOpt)
        {
            _deviceId = deviceId;
        }

        protected override InferenceSession BuildInferenceSession(string modelPath, SessionOptions sessionOptions)
        {
            using SessionOptions options = BuildSessionOptionsBase(sessionOptions);
            options.AppendExecutionProvider_DML(this._deviceId);
            return new InferenceSession(modelPath, options);
        }

        protected override IOcrClassifier CreateOcrClassifier(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess)
        {
            return new TextClassifierIoBinding(session, postprocess, preprocess, OcrConfig, GetDeviceType());
        }

        protected override IOcrDetector CreateOcrDetector(InferenceSession session, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            return new TextDetectorIoBinding(session, postprocess, preprocess, OcrConfig, GetDeviceType());
        }

        protected override IOcrRecognizer CreateOcrRecognizer(InferenceSession session, IRecPostprocess postprocess, IRecPreprocess preprocess)
        {
            return new TextRecognizerIoBinding(session, postprocess, preprocess, OcrConfig, GetDeviceType());
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }
    }
}
