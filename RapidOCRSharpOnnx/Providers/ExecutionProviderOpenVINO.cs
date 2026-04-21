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

        public ExecutionProviderOpenVINO(OcrConfig ocrConfig, IntelDeviceType intelDeviceType) : base(ocrConfig)
        {
            _intelDeviceType = intelDeviceType;
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;
            options.AppendExecutionProvider_OpenVINO(GetIntelDeviceType());

            return options;
        }

        protected override IOcrClassifier CreateOcrClassifier(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new TextClassifierOrtVal(session, options, postprocess, preprocess, OcrConfig);
            }
            else
            {
                return new TextClassifierIoBinding(session, options, postprocess, preprocess, OcrConfig);
            }
        }

        protected override IOcrDetector CreateOcrDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new TextDetectorOrtVal(session, options, postprocess, preprocess);
            }
            else
            {
                return new TextDetectorIoBinding(session, options, postprocess, preprocess);
            }

        }

        protected override IOcrRecognizer CreateOcrRecognizer(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new TextRecognizerOrtVal(session, options, postprocess, preprocess, OcrConfig);
            }
            else
            {
                return new TextRecognizerIoBinding(session, options, postprocess, preprocess, OcrConfig);
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
