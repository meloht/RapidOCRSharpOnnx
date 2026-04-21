using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public abstract class TextRecognizerBase : OnnxInferenceCore
    {

        protected IRecPreprocess _recPreprocess;
        protected IRecPostprocess _recPostprocess;
        protected readonly string[] _charList;

        protected OcrConfig _ocrConfig;


        public TextRecognizerBase(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess, OcrConfig ocrConfig)
            : base(session, options)
        {
            _recPreprocess = preprocess;
            _recPostprocess = postprocess;
            _ocrConfig = ocrConfig;

            var charList = GetCharacterList();
            charList.Insert(0, "blank");
            charList.Add(" ");
            _charList = charList.ToArray();
        }

        private List<string> GetCharacterList(string key = "character")
        {
            var map = _session.ModelMetadata.CustomMetadataMap;
            if (map.ContainsKey(key))
                return map[key].Split('\n', StringSplitOptions.RemoveEmptyEntries).ToList();

            return new List<string>();
        }

        public RecResult[] TextRecognize(Mat[] imgList)
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

            RecResult[] rec_res = new RecResult[imgCount];
            for (int i = 0; i < imgCount; i++)
            {
                rec_res[i] = new RecResult("", 0.0f);
            }
            int img_c = _ocrConfig.RecognizerConfig.RecImgShape[0];
            int img_h = _ocrConfig.RecognizerConfig.RecImgShape[1];
            int img_w = _ocrConfig.RecognizerConfig.RecImgShape[2];

            int idx = 0;

            for (int i = 0, imgIdx = 0; i < imgCount; i += _ocrConfig.RecognizerConfig.RecBatchNum)
            {
                int endNo = Math.Min(imgCount, i + _ocrConfig.RecognizerConfig.RecBatchNum);
                int batchSize = endNo - i;

                float[] wh_ratio_list = new float[batchSize];
                float max_wh_ratio = (float)img_w / (float)img_h;


                for (int j = i, ratioIdx = 0; j < endNo; j++, ratioIdx++)
                {
                    float wh_ratio = (float)imgList[indices[j]].Width / (float)imgList[indices[j]].Height;
                    max_wh_ratio = Math.Max(max_wh_ratio, wh_ratio);
                    wh_ratio_list[ratioIdx] = wh_ratio;
                }

                int img_width = (int)(img_h * max_wh_ratio);
                int tensorLength = img_c * img_h * img_width * batchSize;

                float[] batchData = new float[tensorLength];


                idx = 0;
                for (int j = i; j < endNo; j++)
                {
                    idx = _recPreprocess.ResizeNormImg(imgList[indices[j]], idx, batchData, max_wh_ratio);
                }

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(batchData, new long[] { batchSize, img_c, img_h, img_width });

                System.Diagnostics.Stopwatch sw = Stopwatch.StartNew();
                using var outData = InferenceRun(inputOrtValue);
                using var ortValue = outData[0];
                sw.Stop();
                Console.WriteLine($"Rec Inference time: {sw.ElapsedMilliseconds} ms");

                var res = _recPostprocess.RecPostProcess(ortValue, wh_ratio_list, max_wh_ratio, _charList);

                for (int j = 0; j < res.Length && imgIdx < imgCount; j++, imgIdx++)
                {
                    rec_res[imgIdx] = res[j];
                }
            }

            return rec_res;
        }

        public void DisposeBase()
        {
            _session?.Dispose();
            _options?.Dispose();
            _runOptions?.Dispose();
        }
    }
}
