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
    public abstract class ExecutionProvider : IExecutionProvider
    {

        protected abstract DeviceType GetDeviceType();

        protected abstract SessionOptions BuildSessionOptions();

        protected abstract IOcrDetector CreateOcrDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess);

        protected abstract IOcrClassifier CreateOcrClassifier(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess);
        protected abstract IOcrRecognizer CreateOcrRecognizer(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess);

        public OcrConfig OcrConfig { get; private set; }

        public ExecutionProvider(OcrConfig ocrConfig)
        {
            OcrConfig = ocrConfig;
        }

        public IOcrDetector CreateDetector()
        {
            if (OcrConfig.DetectorConfig == null || string.IsNullOrWhiteSpace(OcrConfig.DetectorConfig.ModelPath))
            {
                throw new ArgumentException("DetectorConfig or ModelPath is null or empty.");
            }
            var options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(OcrConfig.DetectorConfig.ModelPath, options);
            var postprocess = new DetPostprocess(OcrConfig.DetectorConfig);
            var preprocess = new DetPreprocess(OcrConfig);

            return CreateOcrDetector(session, options, postprocess, preprocess);
        }

        public IOcrClassifier CreateClassifier()
        {
            if (OcrConfig.ClassifierConfig == null || string.IsNullOrWhiteSpace(OcrConfig.ClassifierConfig.ModelPath))
            {
                return null;
            }
            var options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(OcrConfig.ClassifierConfig.ModelPath, options);
            var postprocess = new ClsPostprocess(OcrConfig.ClassifierConfig);
            var preprocess = new ClsPreprocess();

            return CreateOcrClassifier(session, options, postprocess, preprocess);
        }

        public IOcrRecognizer CreateRecognizer()
        {
            if (OcrConfig.RecognizerConfig == null || string.IsNullOrWhiteSpace(OcrConfig.RecognizerConfig.ModelPath))
            {
                throw new ArgumentException("RecognizerConfig or ModelPath is null or empty.");
            }
            var options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(OcrConfig.RecognizerConfig.ModelPath, options);
            var postprocess = new RecPostprocess(OcrConfig);
            var preprocess = new RecPreprocess(OcrConfig.RecognizerConfig);

            return CreateOcrRecognizer(session, options, postprocess, preprocess);
        }
    }
}
