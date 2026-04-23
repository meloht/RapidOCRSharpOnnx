using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Flann;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Channels;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public abstract class TextClassifierBase : OnnxInferenceCore
    {

        protected IClsPreprocess _clsPreprocess;
        protected IClsPostprocess _clsPostprocess;

        protected readonly int[] _clsImageShape;

        public TextClassifierBase(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OcrConfig ocrConfig, DeviceType deviceType)
            : base(session, options, ocrConfig, deviceType)
        {
            _clsPreprocess = preprocess;
            _clsPostprocess = postprocess;
            _ocrConfig = ocrConfig;

            _clsImageShape = preprocess.GetClsImageShape();

        }


        public ResultPerf<ClsResult[]> TextClassify(DisposableList<Mat> imgList)
        {
            PerfModel perf = new PerfModel();

            int[] indices = new int[imgList.Count];
            float[] widthList = new float[imgList.Count];
            for (int i = 0; i < indices.Length; i++)
            {
                indices[i] = i;
                widthList[i] = (float)imgList[i].Width / (float)imgList[i].Height;
            }

            Array.Sort(indices, (a, b) => widthList[a].CompareTo(widthList[b]));
            int imgCount = imgList.Count;
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
                _stopwatch.Restart();
                int endNo = Math.Min(imgCount, i + _ocrConfig.ClassifierConfig.ClsBatchNum);
                int batchSize = endNo - i;
                float[] batchData = new float[batchSize * img_c * img_h * img_w];

                idx = 0;
                for (int j = i; j < endNo; j++)
                {
                    idx = _clsPreprocess.ResizeNormImg(imgList[indices[j]], idx, batchData);
                }

                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(batchData, new long[] { batchSize, img_c, img_h, img_w });

                _stopwatch.Stop();
                perf.Preprocess += _stopwatch.ElapsedMilliseconds;


                using var output = InferenceRun(inputOrtValue, perf);

                _stopwatch.Restart();
                using var ortValue = output[0];
                _clsPostprocess.ClsPostProcess(ortValue, i, imgList, cls_res);

                _stopwatch.Stop();
                perf.Postprocess += _stopwatch.ElapsedMilliseconds;

            }
            perf.SumTotal();
            var resultPerf = new ResultPerf<ClsResult[]>();
            resultPerf.Data = cls_res;
            resultPerf.Perf = perf;
            return resultPerf;
        }


        public void BatchClsAsync(OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> recChannelWriter)
        {

            Channel<ClsPreResultBatch> channelPre = Channel.CreateBounded<ClsPreResultBatch>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
            var producer = Task.Run(() => _clsPreprocess.PreprocessBatchAsync(batchResult.DetResult.ImgCropList, _deviceType, batchResult, channelPre.Writer));

            var consumer = WriteRecAsync(batchResult, channelPre, recChannelWriter);

            Task.WaitAll(producer, consumer);
            recChannelWriter.Complete();
            Console.WriteLine($"{DateTime.Now} recChannelWriter.Complete()");

        }
        private async Task WriteRecAsync(OcrBatchResult batchResult, Channel<ClsPreResultBatch> channelPre, ChannelWriter<OcrBatchResult> recChannelWriter)
        {
            int idx = 0;
            int count = batchResult.DetResult.ImgCropList.Count;
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];

            batchResult.ClsResult = new ClsResult[count];

            await foreach (ClsPreResultBatch item in channelPre.Reader.ReadAllAsync())
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(item.InputData, new long[] { 1, img_c, img_h, img_w });
                Console.WriteLine($"{DateTime.Now} Cls batch {idx}");
                var output0 = InferenceRun(inputOrtValue, null);
                //await BatchPostProcessAsync(output0, item.BatchResult, item.img, idx, recChannelWriter);

                using var ortValue = output0[0];
                item.BatchResult.ClsResult[idx] = _clsPostprocess.ClsPostProcess(ortValue, item.img);
                Console.WriteLine($"{DateTime.Now} Cls batch Write {idx}");
                await recChannelWriter.WriteAsync(item.BatchResult);

                Interlocked.Increment(ref idx);

            }

          
        }

        private async Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, OcrBatchResult item, Mat img, int index, ChannelWriter<OcrBatchResult> writer)
        {
            await Task.Run(async () =>
            {
                using (output)
                {
                    using var ortValue = output[0];
                    item.ClsResult[index] = _clsPostprocess.ClsPostProcess(ortValue, img);
                    Console.WriteLine($"Cls batch Write {index}");
                    await writer.WriteAsync(item);
                }
            });
        }

    }
}
