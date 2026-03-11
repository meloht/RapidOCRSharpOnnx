using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.Config
{
    public static class OnnxEngineConfig
    {
        public static int IntraOpNumThreads = -1;
        public static int InterOpNumThreads = -1;
        public static bool EnableCpuMemArena = false;

        public static class CpuEpCfg
        {
            public static string ArenaExtendStrategy = "kSameAsRequested";
        }

        public static bool UseCuda = false;

        public static class CudaEpCfg
        {
            public static int DeviceId = 0;
            public static string ArenaExtendStrategy = "kNextPowerOfTwo";
            public static string CudnnConvAlgoSearch = "EXHAUSTIVE";
            public static bool DoCopyInDefaultStream = true;
        }
        public static bool UseDml = false;
        public static class DmEpCfg
        {
            public static int DeviceId = 0;
        }

        public static bool use_cann = false;

        public static class CannEpCfg
        {
            public static int DeviceId = 0;
            public static string ArenaExtendStrategy = "kNextPowerOfTwo";
            public static long NpuMemLimit = 21474836480;//20 * 1024 * 1024 * 1024
            public static string OpSelectImplMode = "high_performance";
            public static string OptypelistForImplmode = "Gelu";
            public static bool EnableCannGraph = true;
        }
        public static bool UseCoreml = false;

        public static class CoremlEpCfg
        {
            public static string ModelFormat = "MLProgram";
            public static string MLComputeUnits = "ALL";
            public static int RequireStaticInputShapes = 0;
            public static int EnableOnSubgraphs = 0;
            public static string SpecializationStrategy = "FastPrediction";
            public static int ProfileComputePlan = 0;
            public static int AllowLowPrecisionAccumulationOnGPU = 0;
            public static string ModelCacheDirectory = "/tmp/RapidOCR";

        }
    }
}
