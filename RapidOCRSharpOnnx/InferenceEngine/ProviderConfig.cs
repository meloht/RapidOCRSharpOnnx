using Microsoft.ML.OnnxRuntime;
using OpenCvSharp.Internal;
using RapidOCRSharpOnnx.Config;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public class ProviderConfig
    {
        private readonly List<string> _hadProviders;
        private readonly string _defaultProvider;
        public ProviderConfig()
        {
            var providers = OrtEnv.Instance().GetAvailableProviders();
            if (providers == null || providers.Length == 0)
            {
                _hadProviders = [EpNames.CPUExecutionProvider];
            }
            else
            {
                _hadProviders = providers.ToList();
            }
            _defaultProvider = _hadProviders[0];
        }


        public void SetProvider(SessionOptions options)
        {
            switch (OnnxEngineConfig.ExecutionProvider)
            {
                case ExecutionProviders.CUDA_EP:
                    SetCUDAProviderOptions(options);
                    break;
                case ExecutionProviders.DirectML_EP:
                    SetDirectMLProviderOptions(options);
                    break;
             
                case ExecutionProviders.CoreML_EP:
                    SetCoreMLProviderOptions(options);
                    break;
                default:
                    options.AppendExecutionProvider_CPU();
                    break;

            }

        }

        public void SetCUDAProviderOptions(SessionOptions options)
        {
            if (IsCudaAvailable())
            {
                var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally
                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["device_id"] = OnnxEngineConfig.CudaEpCfg.DeviceId.ToString();
                providerOptionsDict["arena_extend_strategy"] = OnnxEngineConfig.CudaEpCfg.ArenaExtendStrategy;
                providerOptionsDict["do_copy_in_default_stream"] = OnnxEngineConfig.CudaEpCfg.DoCopyInDefaultStream ? "1" : "0";

                cudaProviderOptions.UpdateOptions(providerOptionsDict);
                options.AppendExecutionProvider_CUDA(cudaProviderOptions);
            }
            else
            {
                throw new ONNXRuntimeError("CUDA Execution Provider is not available. Please check your environment and configuration.");
            }
        }

        public void SetDirectMLProviderOptions(SessionOptions options)
        {
            if (IsDmlAvailable())
            {
                options.AppendExecutionProvider_DML(OnnxEngineConfig.DmEpCfg.DeviceId);
            }
            else
            {
                throw new ONNXRuntimeError("DirectML Execution Provider is not available. Please check your environment and configuration.");
            }
        }
        public void SetCoreMLProviderOptions(SessionOptions options)
        {
            if (IsCoremlAvailable())
            {
                options.AppendExecutionProvider_CoreML();
            }
            else
            {
                throw new ONNXRuntimeError("CoreML Execution Provider is not available. Please check your environment and configuration.");
            }
        }


        public bool IsCudaAvailable()
        {
            if (OnnxEngineConfig.ExecutionProvider != ExecutionProviders.CUDA_EP)
            {
                return false;
            }
            string providerName = EpNames.CUDAExecutionProvider;
            if (_hadProviders != null && _hadProviders.Contains(providerName))
            {
                return true;
            }
#if DEBUG
            System.Diagnostics.Debug.WriteLine($"{providerName} is not in available providers ({string.Join(", ", _hadProviders)}), Use {_defaultProvider} inference by default.");
#endif

            return false;
        }

        public bool IsDmlAvailable()
        {
            if (OnnxEngineConfig.ExecutionProvider != ExecutionProviders.DirectML_EP)
            {
                return false;
            }

            var isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            if (!isWindows)
            {
                var osInfo = RuntimeInformation.OSDescription;
#if DEBUG
                System.Diagnostics.Debug.WriteLine($"DirectML is only supported in Windows OS. The current OS is {osInfo}. Use {_defaultProvider} inference by default.");
#endif
                return false;
            }

            var windowsBuild = GetWindowsBuildNumber();
            if (windowsBuild < 18362)
            {
#if DEBUG
                System.Diagnostics.Debug.WriteLine($"DirectML is only supported in Windows 10 Build 18362 and above OS. The current Windows Build is {windowsBuild}. Use {_defaultProvider} inference by default.");
#endif
                return false;
            }

            var dmlEp = EpNames.DmlExecutionProvider;
            if (_hadProviders.Contains(dmlEp))
            {
                return true;
            }

#if DEBUG
            System.Diagnostics.Debug.WriteLine($"{dmlEp} is not in available providers ({string.Join(", ", _hadProviders)}). Use {_defaultProvider} inference by default.");
#endif

            return false;
        }


        // 辅助方法：获取Windows Build号（解决Environment.OSVersion在Win10+不准确的问题）
        private int GetWindowsBuildNumber()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return 0;
            }
            return Environment.OSVersion.Version.Build;
        }


        public bool IsCoremlAvailable()
        {
            if (OnnxEngineConfig.ExecutionProvider != ExecutionProviders.CoreML_EP)
            {
                return false;
            }

            var isMacOS = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
            if (!isMacOS)
            {
#if DEBUG
                System.Diagnostics.Debug.WriteLine($"CoreML is only supported in macOS/iOS. The current OS is {RuntimeInformation.OSDescription}. Use {_defaultProvider} inference by default.");
#endif
                return false;
            }

            var coremlEp = EpNames.CoreMLExecutionProvider;
            if (_hadProviders.Contains(coremlEp))
            {
                return true;
            }

#if DEBUG
            System.Diagnostics.Debug.WriteLine($"{coremlEp} is not in available providers ({string.Join(", ", _hadProviders)}). Use {_defaultProvider} inference by default.");
#endif

            return false;
        }

    }
}
