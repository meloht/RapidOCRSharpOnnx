using RapidOCRSharpOnnx.InferenceEngine;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Config
{
    public static class OnnxEngineConfig
    {
        public static int IntraOpNumThreads  = -1;
        public static int InterOpNumThreads  = -1;
        public static bool EnableCpuMemArena  = false;
        public static ExecutionProviders ExecutionProvider= ExecutionProviders.CPU_EP;

        public static class CpuEpCfg
        {
            public static string ArenaExtendStrategy = "kSameAsRequested";
        }

        public static class CudaEpCfg
        {
            public static int DeviceId  = 0;
            public static string ArenaExtendStrategy = "EXHAUSTIVE";
            public static bool DoCopyInDefaultStream = true;
        }

        public static class DmEpCfg
        {
            public static int DeviceId  = 1;
        }
    }
}
