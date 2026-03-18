using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.Utils
{
    public enum DeviceType
    {
        CPU,
        CUDA,
        NPU,
        XPU,
        MLU,
        DCU,
        GCU,
        MPS
    }

    public enum LangDet
    {
        CH,
        EN,
        MULTI
    }
    public enum LangCls
    {
        CH
    }

    public enum LangRec
    {
        CH,
        CH_DOC,
        EN,
        ARABIC,
        CHINESE_CHT,
        CYRILLIC,
        DEVANAGARI,
        JAPAN,
        KOREAN,
        KA,
        LATIN,
        TA,
        TE,
        ESLAV,
        TH,
        EL
    }

    public enum OCRVersion
    {
        PPOCRV4,
        PPOCRV5,
    }

    public enum EngineType
    {
        ONNXRUNTIME
    }

    public enum ModelType
    {
        MOBILE,
        SERVER
    }

    public enum TaskType
    {
        DET,
        CLS,
        REC
    }

    public enum LimitType
    {
        Min,
        Max
    }

    public enum Direction
    {
        HORIZONTAL,
        VERTICAL
    }

    public enum WordType
    {
        CN,
        EN,
        NUM,
        EN_NUM
    }
}
