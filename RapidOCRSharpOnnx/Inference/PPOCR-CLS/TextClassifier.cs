using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public class TextClassifier : IOcrClassifier
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        private IClsPreprocess _clsPreprocess;
        private IClsPostprocess _clsPostprocess;
        private int[] _clsImageShape;
        private OcrConfig _ocrConfig;


        public TextClassifier(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, int[] clsImageShape, OcrConfig ocrConfig)
        {
            _runOptions = new RunOptions();
            _session = session;
            _options = options;
            _clsPreprocess = preprocess;
            _clsPostprocess = postprocess;

            _clsImageShape = clsImageShape;
            _ocrConfig = ocrConfig;
        }


        public InferenceResult[] TextClassify(Mat[] imgList)
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
            InferenceResult[] cls_res = new InferenceResult[imgCount];
            for (int i = 0; i < imgCount; i++)
            {
                cls_res[i] = new InferenceResult("", 0.0f);
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
                    idx = _clsPreprocess.ResizeNormImg(imgList[indices[j]], idx, batchData);
                }

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(batchData, new long[] { batchSize, img_c, img_h, img_w });
                using var runOptions = new RunOptions();

                using var outData = _session.Run(runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
                using var outputOrtValue = outData[0];

                _clsPostprocess.ClsPostProcess(outputOrtValue,i,imgList,cls_res);
               
            }

            return cls_res;
        }



        public void Dispose()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }
    }
}
