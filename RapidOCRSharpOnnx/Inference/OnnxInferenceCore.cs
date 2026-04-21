using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference
{
    public abstract class OnnxInferenceCore
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        protected abstract IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue);

        public OnnxInferenceCore(InferenceSession session, SessionOptions options)
        {
            _runOptions = new RunOptions();
            _session = session;
            _options = options;
        }

        protected IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue, OrtIoBinding binding)
        {
            binding.BindInput(_session.InputNames[0], inputOrtValue);
            binding.BindOutputToDevice(_session.OutputNames[0], OrtMemoryInfo.DefaultInstance);
            binding.SynchronizeBoundInputs();

            var results = _session.RunWithBoundResults(_runOptions, binding);
            binding.SynchronizeBoundOutputs();
            return results;
        }

        protected IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue)
        {
            return _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames); 
        }
    }
}
