using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Det;
using RapidOCRSharpOnnx.Inference.PPOCR_Det.Models;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;


namespace RapidOCRSharpOnnx.Inference
{
    public class ExecutePipeline : IExecutePipeline
    {
        private IExecutionProvider _executionProvider;
        private IOcrDetector _ocrDetector;
        private IOcrClassifier _ocrClassifier;
        private IOcrRecognizer _ocrRecognizer;
        private OcrDrawerSkia _ocrDrawerSkia;
        protected OcrConfig _ocrConfig;

        public ExecutePipeline(OcrConfig ocrConfig, IExecutionProvider executionProvider)
        {
            _ocrConfig = ocrConfig;
            _executionProvider = executionProvider;
            _ocrDetector = _executionProvider.CreateDetector();
            _ocrClassifier = _executionProvider.CreateClassifier();
            _ocrRecognizer = _executionProvider.CreateRecognizer();
            _ocrDrawerSkia = new OcrDrawerSkia(_ocrConfig);
        }

        #region batch channel seq & single thread execute

        public OcrBatchResult[] BatchAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];
            for (int i = 0; i < imageList.Count; i++)
            {
                batchResults[i] = new OcrBatchResult();
                batchResults[i].ImagePath = imageList[i];
            }
            try
            {
                Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
                Channel<OcrBatchResult> channelDetNext = GetDetNextChannel(channelRecPre);

                List<Task> tasks = new List<Task>();
                var task = Task.Run(async () => await _ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));
                tasks.Add(task);
                if (_ocrClassifier != null)
                {
                    var consumerCls = BatchClsRead(channelDetNext, channelRecPre.Writer).ContinueWith(t =>
                    {
                        channelRecPre.Writer.Complete();
                    });
                    tasks.Add(consumerCls);
                }

                var consumerRec = BatchRecRead(channelRecPre, processCallback, receiveAction);

                tasks.Add(consumerRec);

                Task.WaitAll([.. tasks]);

                SaveDrawImage(saveDir, batchResults);
            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                foreach (var item in batchResults)
                {
                    item.DetResult?.ImgCropList?.Dispose();
                }
            }
            return batchResults;
        }

        public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];
            for (int i = 0; i < imageList.Count; i++)
            {
                batchResults[i] = new OcrBatchResult();
                batchResults[i].ImagePath = imageList[i];
            }

            Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));

            Channel<OcrBatchResult> channelDetNext = GetDetNextChannel(channelRecPre);

            _ = Task.Run(async () => await _ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));

            if (_ocrClassifier != null)
            {
                _ = BatchClsRead(channelDetNext, channelRecPre.Writer).ContinueWith(t =>
                {
                    channelRecPre.Writer.Complete();
                });

            }

            CheckSaveDir(saveDir);

            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                await _ocrRecognizer.BatchRecAsync(item);
                _ = InferCompleteAsync(item, processCallback, receiveAction);
                SaveImageWithTextBlocks(item, saveDir);
                item.DetResult?.ImgCropList?.Dispose();
                yield return item;
            }

        }

        public OcrResult RecognizeText(Mat image, string savePath = null)
        {
            OcrResult result = new OcrResult();
            var detResult = _ocrDetector.TextDetect(image);
            result.DetResult = detResult;
            if (detResult.Data.ImgCropList == null || detResult.Data.ImgCropList.Count == 0)
            {
                return result;
            }
            using (detResult.Data.ImgCropList)
            {

                if (_ocrClassifier != null)
                {
                    var ClsResult = _ocrClassifier.TextClassify(detResult.Data.ImgCropList);
                    result.ClsResult = ClsResult;
                }

                var recResults = _ocrRecognizer.TextRecognize(detResult.Data.ImgCropList);
                result.RecResult = recResults;

                for (int i = 0; i < detResult.Data.DetItems.Length; i++)
                {
                    detResult.Data.DetItems[i].Word = recResults.Data[i].Label;
                }
                result.TextBlocks = string.Join(" ", recResults.Data.Select(r => r.Label));

                if (!string.IsNullOrEmpty(savePath))
                {
                    _ocrDrawerSkia.DrawTextBlock(image, savePath, detResult.Data, recResults.Data);
                }
            }

            return result;
        }
        public OcrResult RecognizeTextSeq(Mat image, string savePath = null)
        {
            OcrResult result = new OcrResult();
            var detResult = _ocrDetector.TextDetect(image);
            result.DetResult = detResult;
            if (detResult.Data.ImgCropList == null || detResult.Data.ImgCropList.Count == 0)
            {
                return result;
            }
            using (detResult.Data.ImgCropList)
            {

                if (_ocrClassifier != null)
                {
                    var ClsResult = _ocrClassifier.TextClassifySeq(detResult.Data.ImgCropList);
                    result.ClsResult = ClsResult;
                }

                var recResults = _ocrRecognizer.TextRecognizeSeq(detResult.Data.ImgCropList);
                result.RecResult = recResults;

                for (int i = 0; i < detResult.Data.DetItems.Length; i++)
                {
                    detResult.Data.DetItems[i].Word = recResults.Data[i].Label;
                }
                result.TextBlocks = string.Join(" ", recResults.Data.Select(r => r.Label));

                if (!string.IsNullOrEmpty(savePath))
                {
                    _ocrDrawerSkia.DrawTextBlock(image, savePath, detResult.Data, recResults.Data);
                }
            }

            return result;
        }
        #endregion
        private async Task BatchRecRead(Channel<OcrBatchResult> channelRecPre, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                await _ocrRecognizer.BatchRecAsync(item);
                _ = InferCompleteAsync(item, processCallback, receiveAction);
            }
        }
        private async Task BatchParallelRecRead(Channel<OcrBatchResult> channelRecPre, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                await _ocrRecognizer.BatchParallelRecAsync(item);
                _ = InferCompleteAsync(item, processCallback, receiveAction);
            }
        }
        private async Task InferCompleteAsync(OcrBatchResult result, IBatchProcessCallback processCallback, Action<OcrBatchResult> receiveAction)
        {
            if (processCallback != null)
            {
                await Task.Run(async () =>
                {
                    processCallback.ReceiveProcessResult(result);
                });
            }
            if (receiveAction != null)
            {
                await Task.Run(async () =>
                {
                    receiveAction(result);
                });
            }
        }
        private async Task BatchClsRead(Channel<OcrBatchResult> channel, ChannelWriter<OcrBatchResult> recChannelWriter)
        {
            await foreach (OcrBatchResult item in channel.Reader.ReadAllAsync())
            {
                await _ocrClassifier.BatchClsAsync(item, recChannelWriter);
            }
        }

        private async Task BatchParallelClsRead(Channel<OcrBatchResult> channel, ChannelWriter<OcrBatchResult> recChannelWriter)
        {
            await foreach (OcrBatchResult item in channel.Reader.ReadAllAsync())
            {
                await _ocrClassifier.BatchParallelClsAsync(item, recChannelWriter);
            }
        }

        private Channel<OcrBatchResult> GetDetNextChannel(Channel<OcrBatchResult> channelRecPre)
        {
            if (_ocrClassifier != null)
            {
                return Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
            }
            else
            {
                return channelRecPre;
            }
        }

        #region batch channel Parallel

      
        public OcrBatchResult[] BatchParallelAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];
            for (int i = 0; i < imageList.Count; i++)
            {
                batchResults[i] = new OcrBatchResult();
                batchResults[i].ImagePath = imageList[i];
            }
            try
            {
                Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));
                Channel<OcrBatchResult> channelDetNext = GetDetNextChannel(channelRecPre);

                List<Task> tasks = new List<Task>();
                var task = Task.Run(async () => await _ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));
                tasks.Add(task);
                if (_ocrClassifier != null)
                {
                    var consumerCls = BatchParallelClsRead(channelDetNext, channelRecPre.Writer).ContinueWith(t =>
                    {
                        channelRecPre.Writer.Complete();
                    });

                    tasks.Add(consumerCls);

                }

                var consumerRec = BatchParallelRecRead(channelRecPre, processCallback, receiveAction);

                tasks.Add(consumerRec);

                Task.WaitAll(tasks.ToArray());

                SaveDrawImage(saveDir, batchResults);
            }
            catch (Exception)
            {
                throw;
            }
            finally
            {
                foreach (var item in batchResults)
                {
                    item.DetResult?.ImgCropList?.Dispose();
                }
            }
            return batchResults;
        }


        public async IAsyncEnumerable<OcrBatchResult> BatchParallelForeachAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];
            for (int i = 0; i < imageList.Count; i++)
            {
                batchResults[i] = new OcrBatchResult();
                batchResults[i].ImagePath = imageList[i];
            }

            Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));

            Channel<OcrBatchResult> channelDetNext = GetDetNextChannel(channelRecPre);

            _ = Task.Run(async () => await _ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));

            if (_ocrClassifier != null)
            {
                _ = BatchParallelClsRead(channelDetNext, channelRecPre.Writer).ContinueWith(t =>
                {
                    channelRecPre.Writer.Complete();
                });

            }

            CheckSaveDir(saveDir);

            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                await _ocrRecognizer.BatchParallelRecAsync(item);
                _ = InferCompleteAsync(item, processCallback, receiveAction);
                SaveImageWithTextBlocks(item, saveDir);
                item.DetResult?.ImgCropList?.Dispose();
                yield return item;
            }
        }

        #endregion

        private void SaveDrawImage(string saveDir, OcrBatchResult[] batchResults)
        {
            if (CheckSaveDir(saveDir))
            {
                foreach (var item in batchResults)
                {
                    SaveImageWithTextBlocks(item, saveDir);
                }
            }
        }

        private void SaveImageWithTextBlocks(OcrBatchResult item, string saveDir)
        {
            if (!string.IsNullOrEmpty(saveDir))
            {
                string resPath = $"res_{Path.GetFileName(item.ImagePath)}";
                string savePath = Path.Combine(saveDir, resPath);

                _ocrDrawerSkia.DrawTextBlock(item.ImagePath, savePath, item.DetResult, item.RecResult);
            }
        }
        private bool CheckSaveDir(string saveDir)
        {
            if (!string.IsNullOrEmpty(saveDir))
            {
                if (!Directory.Exists(saveDir))
                {
                    Directory.CreateDirectory(saveDir);
                }
                return true;
            }
            return false;
        }


        public void Dispose()
        {
            _ocrDetector?.Dispose();
            _ocrClassifier?.Dispose();
            _ocrRecognizer?.Dispose();
            _ocrDrawerSkia?.Dispose();
        }
    }
}
