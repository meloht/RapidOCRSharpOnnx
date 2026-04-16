using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Config;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;


namespace RapidOCRSharpOnnx.InferenceEngine
{
    public class OrtInferSession : IDisposable
    {
        private InferenceSession _inferenceSession;
        private SessionOptions _sessionOptions;
        private ProviderConfig _providerConfig;

        public OrtInferSession(string modelPath)
        {
            _providerConfig = new ProviderConfig();
            _sessionOptions = InitSessionOptions();
            _inferenceSession = new InferenceSession(modelPath, _sessionOptions);
        }

        private SessionOptions InitSessionOptions()
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
          //  options.EnableCpuMemArena = OnnxEngineConfig.EnableCpuMemArena;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            int cpu_nums = Environment.ProcessorCount;
            int intra_op_num_threads = OnnxEngineConfig.IntraOpNumThreads;
            if (intra_op_num_threads != -1 && 1 <= intra_op_num_threads && intra_op_num_threads <= cpu_nums)
            {
                options.IntraOpNumThreads = intra_op_num_threads;
            }

            int inter_op_num_threads = OnnxEngineConfig.InterOpNumThreads;
            if (inter_op_num_threads != -1 && 1 <= inter_op_num_threads && inter_op_num_threads <= cpu_nums)
            {
                options.InterOpNumThreads = inter_op_num_threads;
            }

            _providerConfig.SetProvider(options);

            return options;
        }

        public OrtValue RunInference(DataTensorDimensions dataTensor)
        {
            try
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(dataTensor.Data, dataTensor.Dimensions);
                using var runOptions = new RunOptions();
                using var results = _inferenceSession.Run(runOptions, _inferenceSession.InputNames, [inputOrtValue], _inferenceSession.OutputNames);
                var output0 = results[0];
                return output0;
            }
            catch (Exception ex)
            {
                throw new ONNXRuntimeError(ex.Message, ex);
            }
            finally
            {
               // ArrayPool<float>.Shared.Return(dataTensor.Data);
            }

        }

        public string[] GetInputNames()
        {
            var inputNames = _inferenceSession.InputNames;
            return inputNames.ToArray();
        }

        public string[] GetOutputNames()
        {
            var outputNames = _inferenceSession.OutputNames;
            return outputNames.ToArray();
        }

        public List<string> GetCharacterList(string key = "character")
        {
            
            var map = _inferenceSession.ModelMetadata.CustomMetadataMap;
            if (map.ContainsKey(key))
                return map[key].Split('\n').ToList();

            return new List<string>();
        }

        public bool HaveKey(string key= "character")
        {
            var map = _inferenceSession.ModelMetadata.CustomMetadataMap;
            if (map.ContainsKey(key))
                return true;
            return false;
        }


        public void Dispose()
        {
            _sessionOptions?.Dispose();
            _inferenceSession?.Dispose();

        }
    }
}
