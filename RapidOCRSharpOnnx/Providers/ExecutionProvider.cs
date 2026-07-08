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

        protected abstract InferenceSession BuildInferenceSession(string modelPath, SessionOptions sessionOptions);

        protected abstract IOcrDetector CreateOcrDetector(InferenceSession session, IDetPostprocess postprocess, IDetPreprocess preprocess);

        protected abstract IOcrClassifier CreateOcrClassifier(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess);
        protected abstract IOcrRecognizer CreateOcrRecognizer(InferenceSession session, IRecPostprocess postprocess, IRecPreprocess preprocess);

        public OcrConfig OcrConfig { get; private set; }

        private readonly SessionOptions _detOpt;
        private readonly SessionOptions _clsOpt;
        private readonly SessionOptions _recOpt;

        public ExecutionProvider(OcrConfig ocrConfig, SessionOptions detOpt = null, SessionOptions clsOpt = null, SessionOptions recOpt = null)
        {
            OcrConfig = ocrConfig;
            _detOpt = detOpt;
            _clsOpt = clsOpt;
            _recOpt = recOpt;
        }
        protected SessionOptions BuildSessionOptionsBase(SessionOptions sessionOptions)
        {
            if (sessionOptions == null)
            {
                sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                sessionOptions.EnableCpuMemArena = true;
                sessionOptions.EnableMemoryPattern = false;
            }
            return sessionOptions;

        }
        public IOcrDetector CreateDetector()
        {
            if (OcrConfig.DetectorConfig == null || string.IsNullOrWhiteSpace(OcrConfig.DetectorConfig.ModelPath))
            {
                throw new ArgumentException("DetectorConfig or ModelPath is null or empty.");
            }
            InferenceSession session = BuildInferenceSession(OcrConfig.DetectorConfig.ModelPath, _detOpt);
            var postprocess = new DetPostprocess(OcrConfig.DetectorConfig);
            var preprocess = new DetPreprocess(OcrConfig);

            return CreateOcrDetector(session, postprocess, preprocess);
        }

        public IOcrClassifier CreateClassifier()
        {
            if (OcrConfig.ClassifierConfig == null || string.IsNullOrWhiteSpace(OcrConfig.ClassifierConfig.ModelPath))
            {
                return null;
            }
            InferenceSession session = BuildInferenceSession(OcrConfig.ClassifierConfig.ModelPath, _clsOpt);
            var postprocess = new ClsPostprocess(OcrConfig.ClassifierConfig);
            var preprocess = new ClsPreprocess(OcrConfig);

            return CreateOcrClassifier(session, postprocess, preprocess);
        }

        public IOcrRecognizer CreateRecognizer()
        {
            if (OcrConfig.RecognizerConfig == null || string.IsNullOrWhiteSpace(OcrConfig.RecognizerConfig.ModelPath))
            {
                throw new ArgumentException("RecognizerConfig or ModelPath is null or empty.");
            }
            InferenceSession session = BuildInferenceSession(OcrConfig.RecognizerConfig.ModelPath, _recOpt);
            var postprocess = new RecPostprocess(OcrConfig);
            var preprocess = new RecPreprocess(OcrConfig.RecognizerConfig);

            return CreateOcrRecognizer(session, postprocess, preprocess);
        }
    }
}
