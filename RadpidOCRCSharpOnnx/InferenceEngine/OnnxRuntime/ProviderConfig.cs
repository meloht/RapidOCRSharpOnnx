using Microsoft.ML.OnnxRuntime;
using OpenCvSharp.Internal;
using RadpidOCRCSharpOnnx.Config;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine.OnnxRuntime
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
                _hadProviders = ["CPUExecutionProvider"];
            }
            else
            {
                _hadProviders = providers.ToList();
            }
            _defaultProvider = _hadProviders[0];

        }


        public List<string> GetProviderList()
        {
            return null;
        }

        public bool IsCudaAvailable()
        {
            if (!OnnxEngineConfig.UseCuda)
            {
                return false;
            }
            string providerName = ExecutionProviders.CUDA_EP.GetDescription();
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
            if (!OnnxEngineConfig.UseDml)
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

            var dmlEp = ExecutionProviders.DIRECTML_EP.GetDescription();
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

    }
}
