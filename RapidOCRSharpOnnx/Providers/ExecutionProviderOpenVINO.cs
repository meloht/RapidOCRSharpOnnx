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
    public class ExecutionProviderOpenVINO : ExecutionProvider
    {
        private const string CPU = "CPU";
        private const string GPU = "GPU";
        private const string GPU0 = "GPU.0";
        private const string GPU1 = "GPU.1";
        private const string NPU = "NPU";
        private IntelDeviceType _intelDeviceType;

        public ExecutionProviderOpenVINO(OcrConfig ocrConfig, IntelDeviceType intelDeviceType)
            : this(ocrConfig, intelDeviceType, null, null, null)
        {
        }
        public ExecutionProviderOpenVINO(OcrConfig ocrConfig, IntelDeviceType intelDeviceType, SessionOptions detOpt = null, SessionOptions clsOpt = null, SessionOptions recOpt = null)
            : base(ocrConfig, detOpt, clsOpt, recOpt)
        {
            _intelDeviceType = intelDeviceType;
        }

        protected override InferenceSession BuildInferenceSession(string modelPath, SessionOptions sessionOptions)
        {
            using SessionOptions options = BuildSessionOptionsBase(sessionOptions);

            options.AppendExecutionProvider_OpenVINO(GetIntelDeviceType());

            return new InferenceSession(modelPath, options);
        }

        protected override IOcrClassifier CreateOcrClassifier(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new TextClassifierOrtVal(session, postprocess, preprocess, OcrConfig, GetDeviceType());
            }
            else
            {
                return new TextClassifierIoBinding(session, postprocess, preprocess, OcrConfig, GetDeviceType());
            }
        }

        protected override IOcrDetector CreateOcrDetector(InferenceSession session, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new TextDetectorOrtVal(session, postprocess, preprocess, OcrConfig, GetDeviceType());
            }
            else
            {
                return new TextDetectorIoBinding(session, postprocess, preprocess, OcrConfig, GetDeviceType());
            }

        }

        protected override IOcrRecognizer CreateOcrRecognizer(InferenceSession session, IRecPostprocess postprocess, IRecPreprocess preprocess)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new TextRecognizerOrtVal(session, postprocess, preprocess, OcrConfig, GetDeviceType());
            }
            else
            {
                return new TextRecognizerIoBinding(session, postprocess, preprocess, OcrConfig, GetDeviceType());
            }

        }

        protected override DeviceType GetDeviceType()
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return DeviceType.CPU;
            }
            else if (_intelDeviceType == IntelDeviceType.NPU)
            {
                return DeviceType.NPU;
            }
            return DeviceType.GPU;
        }

        private string GetIntelDeviceType()
        {
            switch (_intelDeviceType)
            {
                case IntelDeviceType.CPU:
                    return CPU;
                case IntelDeviceType.GPU:
                    return GPU;
                case IntelDeviceType.GPU0:
                    return GPU0;
                case IntelDeviceType.GPU1:
                    return GPU1;
                case IntelDeviceType.NPU:
                    return NPU;
                default:
                    return CPU;
            }
        }
    }
}
