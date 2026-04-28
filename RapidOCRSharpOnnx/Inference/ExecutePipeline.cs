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

        public OcrBatchResult[] BatchAsync(List<string> imageList, string saveDir = null)
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
                var task = Task.Run(() => _ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));
                tasks.Add(task);
                if (_ocrClassifier != null)
                {
                    var consumerCls = BatchClsRead(channelDetNext, channelRecPre.Writer);
                    consumerCls.ContinueWith(t =>
                    {
                        channelRecPre.Writer.Complete();
                        //Console.WriteLine($"{DateTime.Now.ToString("HH:mm:ss.fff")} recChannelWriter.Complete()");
                    });
                    tasks.Add(consumerCls);
                }

                var consumerRec = BatchRecRead(channelRecPre);

                tasks.Add(consumerRec);

                Task.WaitAll(tasks.ToArray());

                if (CheckSaveDir(saveDir))
                {
                    foreach (var item in batchResults)
                    {
                        SaveImageWithTextBlocks(item, saveDir);
                    }
                }
            }
            catch (Exception ex)
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

        private async Task BatchRecRead(Channel<OcrBatchResult> channelRecPre)
        {
            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                _ocrRecognizer.BatchRecAsync(item);
            }
        }

        private async Task BatchClsRead(Channel<OcrBatchResult> channel, ChannelWriter<OcrBatchResult> recChannelWriter)
        {
            await foreach (OcrBatchResult item in channel.Reader.ReadAllAsync())
            {
                _ocrClassifier.BatchClsAsync(item, recChannelWriter);
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

        public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList, string saveDir = null)
        {
            OcrBatchResult[] batchResults = new OcrBatchResult[imageList.Count];
            for (int i = 0; i < imageList.Count; i++)
            {
                batchResults[i] = new OcrBatchResult();
                batchResults[i].ImagePath = imageList[i];
            }

            Channel<OcrBatchResult> channelRecPre = Channel.CreateBounded<OcrBatchResult>(UtilsHelper.GetChannelOptions(_ocrConfig.BatchPoolSize));

            Channel<OcrBatchResult> channelDetNext = GetDetNextChannel(channelRecPre);

            _ = Task.Run(() => _ocrDetector.BatchDetectAsync(imageList, channelDetNext.Writer, batchResults));

            if (_ocrClassifier != null)
            {
                var consumerCls = BatchClsRead(channelDetNext, channelRecPre.Writer);
                _ = consumerCls.ContinueWith(t =>
                 {
                     channelRecPre.Writer.Complete();
                     //Console.WriteLine($"{DateTime.Now.ToString("HH:mm:ss.fff")} recChannelWriter.Complete()");
                 });
                _ = Task.Run(() => consumerCls);

            }

            CheckSaveDir(saveDir);

            await foreach (OcrBatchResult item in channelRecPre.Reader.ReadAllAsync())
            {
                _ocrRecognizer.BatchRecAsync(item);
                SaveImageWithTextBlocks(item, saveDir);
                item.DetResult?.ImgCropList?.Dispose();
                yield return item;
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

        public OcrResult RecognizeText(string imagePath, string savePath = null)
        {
            using Mat image = Cv2.ImRead(imagePath);
            return RecognizeText(image, savePath);
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
        public OcrResult RecognizeTextSeq(string imagePath, string savePath = null)
        {
            using Mat image = Cv2.ImRead(imagePath);
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
        public void Dispose()
        {
            _ocrDetector?.Dispose();
            _ocrClassifier?.Dispose();
            _ocrRecognizer?.Dispose();
            _ocrDrawerSkia?.Dispose();
        }
    }
}
