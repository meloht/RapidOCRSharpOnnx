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
    public class ExecutionProviderCUDA : ExecutionProvider
    {
        private int _deviceId;
        private Dictionary<string, string> _providerOptionsDict;

        public ExecutionProviderCUDA(OcrConfig ocrConfig, int deviceId = 0, Dictionary<string, string> providerOptionsDict = null) 
            : this(ocrConfig, deviceId, providerOptionsDict, null, null, null)
        {
        }
        public ExecutionProviderCUDA(OcrConfig ocrConfig, int deviceId = 0, Dictionary<string, string> providerOptionsDict = null, SessionOptions detOpt = null, SessionOptions clsOpt = null, SessionOptions recOpt = null)
            : base(ocrConfig, detOpt, clsOpt, recOpt)
        {
            _deviceId = deviceId;
            _providerOptionsDict = providerOptionsDict;
        }

        protected override InferenceSession BuildInferenceSession(string modelPath, SessionOptions sessionOptions)
        {
            using SessionOptions options = BuildSessionOptionsBase(sessionOptions);
            if (this._providerOptionsDict != null && this._providerOptionsDict.Count > 0)
            {
                if (_providerOptionsDict.ContainsKey("device_id"))
                {
                    _providerOptionsDict["device_id"] = _deviceId.ToString();
                }
                else
                {
                    _providerOptionsDict.Add("device_id", _deviceId.ToString());
                }
                using var cudaProviderOptions = new OrtCUDAProviderOptions();
                cudaProviderOptions.UpdateOptions(_providerOptionsDict);
                options.AppendExecutionProvider_CUDA(cudaProviderOptions);
            }
            else
            {
                options.AppendExecutionProvider_CUDA(_deviceId);
            }

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
