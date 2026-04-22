using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference
{
    public abstract class OnnxInferenceCore
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;
        protected Stopwatch _stopwatch;
        protected readonly DeviceType _deviceType;
        protected OcrConfig _ocrConfig;
        protected abstract IDisposableReadOnlyCollection<OrtValue> InferenceRun(OrtValue inputOrtValue, PerfModel perf);

        public OnnxInferenceCore(InferenceSession session, SessionOptions options, OcrConfig ocrConfig, DeviceType deviceType)
        {
            _stopwatch = new Stopwatch();
            _runOptions = new RunOptions();
            _session = session;
            _options = options;
            _deviceType = deviceType;
            _ocrConfig = ocrConfig;
        }

        protected IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue, OrtIoBinding binding, PerfModel perf)
        {
            if(perf == null)
            {
                return InferenceRunCore(inputOrtValue, binding);
            }
            _stopwatch.Restart();

            var results = InferenceRunCore(inputOrtValue, binding);

            _stopwatch.Stop();
            perf.Inference += _stopwatch.ElapsedMilliseconds;
            return results;
        }
        private IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue, OrtIoBinding binding)
        {
            binding.BindInput(_session.InputNames[0], inputOrtValue);
            binding.BindOutputToDevice(_session.OutputNames[0], OrtMemoryInfo.DefaultInstance);
            binding.SynchronizeBoundInputs();

            var results = _session.RunWithBoundResults(_runOptions, binding);
            binding.SynchronizeBoundOutputs();
            return results;
        }

        protected IDisposableReadOnlyCollection<OrtValue> InferenceRunCore(OrtValue inputOrtValue, PerfModel perf)
        {
            if(perf == null)
            {
                return _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames); 
            }
            _stopwatch.Restart();
            var res = _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);

            _stopwatch.Stop();
            perf.Inference += _stopwatch.ElapsedMilliseconds;
            return res;
        }


        protected BoundedChannelOptions GetChannelOptions(int batchPoolSize)
        {
            var channelOptions = new BoundedChannelOptions(batchPoolSize)
            {
                SingleWriter = false,
                SingleReader = true,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.Wait
            };
            
            return channelOptions;
        }

        public void DisposeBase()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }
    }
}
