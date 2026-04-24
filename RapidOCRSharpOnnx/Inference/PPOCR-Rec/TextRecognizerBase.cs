using Clipper2Lib;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Flann;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public abstract class TextRecognizerBase : OnnxInferenceCore
    {

        protected IRecPreprocess _recPreprocess;
        protected IRecPostprocess _recPostprocess;
        protected readonly string[] _charList;


        public TextRecognizerBase(InferenceSession session, SessionOptions options, IRecPostprocess postprocess, IRecPreprocess preprocess, OcrConfig ocrConfig, DeviceType deviceType)
            : base(session, options, ocrConfig, deviceType)
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

        public ResultPerf<RecResult[]> TextRecognize(DisposableList<ImageIndex> imgList)
        {
            PerfModel perf = new PerfModel();

            float[] widthList = new float[imgList.Count];

            int imgCount = imgList.Count;

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
                _stopwatch.Restart();
                int endNo = Math.Min(imgCount, i + _ocrConfig.RecognizerConfig.RecBatchNum);
                int batchSize = endNo - i;

                float[] wh_ratio_list = new float[batchSize];

                float config_wh_ratio = (float)img_w / (float)img_h;
                float[] max_wh_ratio_list = new float[batchSize];
                float max_wh_ratio = config_wh_ratio;
                for (int j = i, ratioIdx = 0; j < endNo; j++, ratioIdx++)
                {
                    float wh_ratio = (float)imgList[j].Image.Width / (float)imgList[j].Image.Height;

                    wh_ratio_list[ratioIdx] = wh_ratio;
                    max_wh_ratio_list[ratioIdx] = Math.Max(config_wh_ratio, wh_ratio);
                    max_wh_ratio = Math.Max(max_wh_ratio, max_wh_ratio_list[ratioIdx]);
                }

                int img_width = (int)Math.Round(img_h * max_wh_ratio, 0);
                int tensorLength = img_c * img_h * img_width * batchSize;

                float[] batchData = new float[tensorLength];


                idx = 0;
                for (int j = i, ratioIdx = 0; j < endNo; j++, ratioIdx++)
                {
                    int img_max_width = (int)Math.Round(img_h * max_wh_ratio_list[ratioIdx], 0);
                    idx = _recPreprocess.ResizeNormImg(imgList[j].Image, idx, batchData, img_width, img_max_width);
                }

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(batchData, new long[] { batchSize, img_c, img_h, img_width });

                _stopwatch.Stop();
                perf.Preprocess += _stopwatch.ElapsedMilliseconds;

                using var outData = InferenceRun(inputOrtValue, perf);
                _stopwatch.Restart();

                using var ortValue = outData[0];
                var res = _recPostprocess.RecPostProcess(ortValue, wh_ratio_list, max_wh_ratio, _charList);

                for (int j = 0; j < res.Length && imgIdx < imgCount; j++, imgIdx++)
                {
                    rec_res[imgIdx] = res[j];
                }

                _stopwatch.Stop();
                perf.Postprocess += _stopwatch.ElapsedMilliseconds;
            }
            perf.SumTotal();
            var resultPerf = new ResultPerf<RecResult[]>();
            resultPerf.Data = rec_res;
            resultPerf.Perf = perf;
            return resultPerf;
        }

        public ResultPerf<RecResult[]> TextRecognizeSeq(DisposableList<ImageIndex> imgList)
        {
            PerfModel perf = new PerfModel();
            RecResult[] rec_res = new RecResult[imgList.Count];
            int img_c = _ocrConfig.RecognizerConfig.RecImgShape[0];
            int img_h = _ocrConfig.RecognizerConfig.RecImgShape[1];
            int img_w = _ocrConfig.RecognizerConfig.RecImgShape[2];

            foreach (ImageIndex imgIdx in imgList)
            {
                _stopwatch.Restart();

                var pre = _recPreprocess.PreprocessSeq(imgIdx);

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(pre.InputData, new long[] { 1, img_c, img_h, pre.ImgWidth });

                _stopwatch.Stop();
                perf.Preprocess += _stopwatch.ElapsedMilliseconds;

                using var outData = InferenceRun(inputOrtValue, perf);
                _stopwatch.Restart();

                using var ortValue = outData[0];
                rec_res[imgIdx.Index] = _recPostprocess.RecPostProcess(ortValue, pre.WhRatio, pre.MaxWhRatio, _charList);

            }

            perf.SumTotal();
            var resultPerf = new ResultPerf<RecResult[]>();
            resultPerf.Data = rec_res;
            resultPerf.Perf = perf;
            return resultPerf;
        }
        public void BatchRecAsync(OcrBatchResult batchResult)
        {

            int count = batchResult.DetResult.ImgCropList.Count;
            batchResult.RecResult = new RecResult[count];
            Channel<RecPreResultBatch> channelPre = Channel.CreateBounded<RecPreResultBatch>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
            var producer = Task.Run(() => _recPreprocess.PreprocessBatchAsync(batchResult.DetResult.ImgCropList, _deviceType, channelPre.Writer));

            var consumer = ForeachReadAsync(channelPre, batchResult);

            Task.WaitAll(producer, consumer);

            for (int i = 0; i < batchResult.DetResult.DetItems.Length && i < batchResult.RecResult.Length; i++)
            {
                batchResult.DetResult.DetItems[i].Word = batchResult.RecResult[i].Label;
            }
            batchResult.TextBlocks = string.Join(" ", batchResult.RecResult.Select(r => r.Label));


        }

        private async Task ForeachReadAsync(Channel<RecPreResultBatch> channelPre, OcrBatchResult batchResult)
        {

            int img_c = _ocrConfig.RecognizerConfig.RecImgShape[0];
            int img_h = _ocrConfig.RecognizerConfig.RecImgShape[1];

            await foreach (RecPreResultBatch item in channelPre.Reader.ReadAllAsync())
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(item.InputData, new long[] { 1, img_c, img_h, item.ImgWidth });
                Console.WriteLine($"Rec batch {item.Index}");
                var output0 = InferenceRun(inputOrtValue, null);
                // await BatchPostProcessAsync(output0, item.BatchResult, item.WhRatio, item.MaxWhRatio, idx);

                using (output0)
                {
                    using var ortValue = output0[0];
                    batchResult.RecResult[item.Index] = _recPostprocess.RecPostProcess(ortValue, item.WhRatio, item.MaxWhRatio, _charList);

                    Console.WriteLine($"Rec RecPostProcess {item.Index}");
                }

            }
        }

        private async Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, OcrBatchResult item, float wh_ratio, float max_wh_ratio, int index)
        {
            await Task.Run(async () =>
            {
                using (output)
                {
                    using var ortValue = output[0];
                    item.RecResult[index] = _recPostprocess.RecPostProcess(ortValue, wh_ratio, max_wh_ratio, _charList);

                    Console.WriteLine($"Rec RecPostProcess {index}");
                }
            });

        }

    }
}
