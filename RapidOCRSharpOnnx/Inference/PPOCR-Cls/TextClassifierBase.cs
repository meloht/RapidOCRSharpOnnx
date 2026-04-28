using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Flann;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Cls.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Channels;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public abstract class TextClassifierBase : OnnxInferenceCore
    {

        protected IClsPreprocess _clsPreprocess;
        protected IClsPostprocess _clsPostprocess;

        protected readonly FixedBuffer _inputFixedBuffer;
        protected OrtValue _inputOrtValue;
        protected Mat _resizedImg;
        protected readonly int[] _clsImageShape;
        private readonly object _detectLock = new();
        protected MatBufferPool _matPool;
        private int _batchPoolSize = 0;
        private long[] _inputShape;
        private int _inputShapeSize;
        private int _inputSizeInBytes;

        public TextClassifierBase(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OcrConfig ocrConfig, DeviceType deviceType)
            : base(session, options, ocrConfig, deviceType)
        {
            _clsPreprocess = preprocess;
            _clsPostprocess = postprocess;
            _ocrConfig = ocrConfig;

            _clsImageShape = preprocess.GetClsImageShape();
            _inputShape = [1, _clsImageShape[0], _clsImageShape[1], _clsImageShape[2]];
            _inputShapeSize = _clsImageShape[0] * _clsImageShape[1] * _clsImageShape[2];

            _inputSizeInBytes = sizeof(float) * _inputShapeSize;
            _inputFixedBuffer = new FixedBuffer(_inputShapeSize);
            _resizedImg = new Mat();
            _inputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float, _inputShape, _inputFixedBuffer.Address, _inputSizeInBytes);


        }

        public void InitBufferPool(int batchPoolSize)
        {
            if (batchPoolSize != _batchPoolSize)
            {
                lock (_detectLock)
                {
                    if (batchPoolSize != _batchPoolSize)
                    {
                        _matPool?.Dispose();
                        _matPool = null;
                        _batchPoolSize = batchPoolSize;
                    }
                }
            }

            if (_matPool == null)
            {
                lock (_detectLock)
                {
                    if (_matPool == null)
                    {
                        _matPool = new MatBufferPool(batchPoolSize, _inputSizeInBytes, _inputShape);
                    }
                }
            }
        }

        protected void DisposeClsBase()
        {
            _inputFixedBuffer.Dispose();
            _inputOrtValue.Dispose();
            _resizedImg.Dispose();
            _matPool?.Dispose();
        }
        public ResultPerf<ClsResult[]> TextClassify(DisposableList<ImageIndex> imgList)
        {
            PerfModel perf = new PerfModel();

            float[] widthList = new float[imgList.Count];

            int imgCount = imgList.Count;
            ClsResult[] cls_res = new ClsResult[imgCount];
            for (int i = 0; i < imgCount; i++)
            {
                cls_res[i] = new ClsResult("", 0.0f);
            }
            int img_c = _clsImageShape[0];
            int img_h = _clsImageShape[1];
            int img_w = _clsImageShape[2];
            long[] inputShape = [1, img_c, img_h, img_w];

            for (int batchIndex = 0; batchIndex < imgCount; batchIndex += _ocrConfig.ClassifierConfig.ClsBatchNum)
            {
                _stopwatch.Restart();
                int endNo = Math.Min(imgCount, batchIndex + _ocrConfig.ClassifierConfig.ClsBatchNum);
                int batchSize = endNo - batchIndex;
                int len = batchSize * _inputShapeSize;
                inputShape[0] = batchSize;

                float[] batchData = ArrayPool<float>.Shared.Rent(len);
                IDisposableReadOnlyCollection<OrtValue> outData = null;
                try
                {
                    int idx = batchIndex;
                    Parallel.For(batchIndex, endNo, _parallelOptions, j =>
                    {
                        using Mat reszieImg = new Mat();
                        _clsPreprocess.ResizeNormImg(imgList[j].Image, j - idx, reszieImg, batchData);
                    });

                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(batchData, inputShape);

                    _stopwatch.Stop();
                    perf.Preprocess += _stopwatch.ElapsedMilliseconds;

                    outData = InferenceRun(inputOrtValue, perf);

                    _stopwatch.Restart();
                }
                catch (Exception)
                {
                    throw;
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(batchData, true);
                }

                if (outData != null)
                {
                    using (outData)
                    {
                        using var ortValue = outData[0];
                        _clsPostprocess.ClsPostProcess(ortValue, batchIndex, imgList, cls_res);

                        _stopwatch.Stop();
                        perf.Postprocess += _stopwatch.ElapsedMilliseconds;
                    }
                }
            }
            perf.SumTotal();
            var resultPerf = new ResultPerf<ClsResult[]>();
            resultPerf.Data = cls_res;
            resultPerf.Perf = perf;
            return resultPerf;
        }
        public ResultPerf<ClsResult[]> TextClassifySeq(DisposableList<ImageIndex> imgList)
        {
            PerfModel perf = new PerfModel();
            ClsResult[] results = new ClsResult[imgList.Count];

            foreach (var item in imgList)
            {
                _stopwatch.Restart();
               
                _clsPreprocess.ResizeNormImg(item.Image, _resizedImg, _inputFixedBuffer);

                _stopwatch.Stop();
                perf.Preprocess += _stopwatch.ElapsedMilliseconds;

                using var output = InferenceRun(_inputOrtValue, perf);

                _stopwatch.Restart();
                using var ortValue = output[0];
                results[item.Index] = _clsPostprocess.ClsPostProcess(ortValue, item.Image);

                _stopwatch.Stop();
                perf.Postprocess += _stopwatch.ElapsedMilliseconds;
            }

            perf.SumTotal();
            var resultPerf = new ResultPerf<ClsResult[]>();
            resultPerf.Data = results;
            resultPerf.Perf = perf;
            return resultPerf;
        }



        public void BatchClsAsync(OcrBatchResult batchResult, ChannelWriter<OcrBatchResult> recChannelWriter)
        {
            InitBufferPool(_ocrConfig.BatchPoolSize);

            int count = batchResult.DetResult.ImgCropList.Count;
            batchResult.ClsResult = new ClsResult[count];
            Channel<ClsPreResultBatch> channelPre = Channel.CreateBounded<ClsPreResultBatch>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));

            var producer = Task.Run(() => _clsPreprocess.PreprocessBatchAsync(batchResult.DetResult.ImgCropList, _matPool, _deviceType, channelPre.Writer));

            var consumer = WriteRecAsync(batchResult, channelPre, recChannelWriter);

            Task.WaitAll(producer, consumer);


        }
        private async Task WriteRecAsync(OcrBatchResult batchResult, Channel<ClsPreResultBatch> channelPre, ChannelWriter<OcrBatchResult> recChannelWriter)
        {

            int count = batchResult.DetResult.ImgCropList.Count;
            Task[] producer = new Task[count];
            int idx = 0;
            await foreach (ClsPreResultBatch item in channelPre.Reader.ReadAllAsync())
            {
                Console.WriteLine($"{DateTime.Now} Cls batch {item.ImageIndex.Index}");
                var output0 = InferenceRun(item.InputData.InputOrtValue, null);
                _matPool.Return(item.InputData);
                producer[idx] = BatchPostProcessAsync(output0, batchResult, item);
                Interlocked.Increment(ref idx);
            }
            await Task.WhenAll(producer);
            await recChannelWriter.WriteAsync(batchResult);

        }

        private Task BatchPostProcessAsync(IDisposableReadOnlyCollection<OrtValue> output, OcrBatchResult batchResult, ClsPreResultBatch item)
        {
            return Task.Run(async () =>
            {
                using (output)
                {
                    using var ortValue = output[0];
                    batchResult.ClsResult[item.ImageIndex.Index] = _clsPostprocess.ClsPostProcess(ortValue, item.ImageIndex.Image);
                    Console.WriteLine($"Cls batch Write {item.ImageIndex.Index}");

                }
            });
        }

    }
}
