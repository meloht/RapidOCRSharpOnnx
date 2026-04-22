using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace RapidOCRSharpOnnx.Inference
{
    public abstract class OnnxInferenceCore
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;
        protected Stopwatch _stopwatch;
        protected abstract IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue, PerfModel perf);

        public OnnxInferenceCore(InferenceSession session, SessionOptions options)
        {
            _stopwatch = new Stopwatch();
            _runOptions = new RunOptions();
            _session = session;
            _options = options;
        }

        protected IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue, OrtIoBinding binding, PerfModel perf)
        {
            _stopwatch.Restart();

            binding.BindInput(_session.InputNames[0], inputOrtValue);
            binding.BindOutputToDevice(_session.OutputNames[0], OrtMemoryInfo.DefaultInstance);
            binding.SynchronizeBoundInputs();

            var results = _session.RunWithBoundResults(_runOptions, binding);
            binding.SynchronizeBoundOutputs();
            _stopwatch.Stop();
            perf.Inference += _stopwatch.ElapsedMilliseconds;
            return results;
        }

        protected IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue, PerfModel perf)
        {
            _stopwatch.Restart();
            var res= _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);

            _stopwatch.Stop();
            perf.Inference += _stopwatch.ElapsedMilliseconds;
            return res;
        }

        public void DisposeBase()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }
    }
}
