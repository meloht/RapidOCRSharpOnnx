using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public abstract class TextClassifierBase : OnnxInferenceCore
    {

        protected IClsPreprocess _clsPreprocess;
        protected IClsPostprocess _clsPostprocess;

        protected OcrConfig _ocrConfig;
        private static readonly int[] ClsImageShapev4 = [3, 48, 192];
        private static readonly int[] ClsImageShapev5 = [3, 80, 160];

        protected readonly int[] _clsImageShape;


        public TextClassifierBase(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OcrConfig ocrConfig)
            : base(session, options)
        {
            _clsPreprocess = preprocess;
            _clsPostprocess = postprocess;
            _ocrConfig = ocrConfig;

            if (_ocrConfig.ClassifierConfig.OCRVersion == OCRVersion.PPOCRV5)
            {
                _clsImageShape = ClsImageShapev5;
            }
            else
            {
                _clsImageShape = ClsImageShapev4;
            }
        }


        public ClsResult[] TextClassify(Mat[] imgList)
        {
            int[] indices = new int[imgList.Length];
            float[] widthList = new float[imgList.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                indices[i] = i;
                widthList[i] = (float)imgList[i].Width / (float)imgList[i].Height;
            }

            Array.Sort(indices, (a, b) => widthList[a].CompareTo(widthList[b]));
            int imgCount = imgList.Length;
            ClsResult[] cls_res = new ClsResult[imgCount];
            for (int i = 0; i < imgCount; i++)
            {
                cls_res[i] = new ClsResult("", 0.0f);
            }
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];

            int idx = 0;
            for (int i = 0; i < imgCount; i += _ocrConfig.ClassifierConfig.ClsBatchNum)
            {
                int endNo = Math.Min(imgCount, i + _ocrConfig.ClassifierConfig.ClsBatchNum);
                int batchSize = endNo - i;
                float[] batchData = new float[batchSize * img_c * img_h * img_w];


                idx = 0;
                for (int j = i; j < endNo; j++)
                {
                    idx = _clsPreprocess.ResizeNormImg(imgList[indices[j]], idx, batchData, _clsImageShape);
                }

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(batchData, new long[] { batchSize, img_c, img_h, img_w });
                using var output = InferenceRun(inputOrtValue);
                using var ortValue = output[0];
                _clsPostprocess.ClsPostProcess(ortValue, i, imgList, cls_res);

            }

            return cls_res;
        }


        public void DisposeBase()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }

    }
}
