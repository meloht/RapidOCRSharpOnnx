using OpenCvSharp;
using OpenCvSharp.ML;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx
{
    public class RapidOCRSharp : IDisposable
    {
        private IExecutionProvider _executionProvider;
        private IExecutePipeline _executePipeline;


        public OcrConfig Configuration
        {
            get { return _executionProvider.OcrConfig; }
        }


        public RapidOCRSharp(IExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
            _executePipeline = new ExecutePipeline(Configuration, executionProvider);
        }
        /// <summary>
        /// Single-threaded inference can set the inference batchsize(ClassifierConfig.ClsBatchNum & RecognizerConfig.RecBatchNum), if the batch size is too large, out of memory will occur
        /// </summary>
        /// <param name="imagePath">image full path</param>
        /// <param name="savePath">the save path for draw result image</param>
        /// <returns>Result including detect result & classifier result(optional) recognize Result  & performance time</returns>
        public OcrResult RecognizeText(string imagePath, string savePath = null)
        {
            ValidationUtils.ValidateImage(imagePath);
            using var img = Cv2.ImRead(imagePath);
            return _executePipeline.RecognizeText(img, savePath);
        }
        /// <summary>
        /// Single-threaded inference can set the inference batchsize(ClassifierConfig.ClsBatchNum & RecognizerConfig.RecBatchNum), if the batch size is too large, out of memory will occur
        /// </summary>
        /// <param name="image">opencv mat 3 channel color image</param>
        /// <param name="savePath">the save path for draw result image(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize Result  & performance time</returns>
        public OcrResult RecognizeText(Mat image, string savePath = null)
        {
            return _executePipeline.RecognizeText(image, savePath);
        }
        /// <summary>
        /// Single-threaded inference cannot set the inference batchsize( default batchsize = 1 )
        /// </summary>
        /// <param name="imagePath">image full path</param>
        /// <param name="savePath">the save path for draw result image(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize Result  & performance time</returns>
        public OcrResult RecognizeTextSeq(string imagePath, string savePath = null)
        {
            ValidationUtils.ValidateImage(imagePath);
            using var img = Cv2.ImRead(imagePath);
            return _executePipeline.RecognizeTextSeq(img, savePath);
        }

        /// <summary>
        /// Single-threaded inference cannot set the inference batchsize( default batchsize = 1 )
        /// </summary>
        /// <param name="image">opencv mat 3 channel color image</param>
        /// <param name="savePath">the save path for draw result image(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize Result  & performance time</returns>
        public OcrResult RecognizeTextSeq(Mat image, string savePath = null)
        {
            return _executePipeline.RecognizeTextSeq(image, savePath);
        }
        /// <summary>
        /// Muti channel batch inference cannot set the inference batchsize( default batchsize = 1 )
        /// </summary>
        /// <param name="dir">images folder</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public OcrBatchResult[] BatchAsync(string dir, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = ValidationUtils.ValidationImageBatch(dir);
            return BatchAsync(list, saveDir, processCallback, receiveAction);
        }
        /// <summary>
        /// Muti channel batch inference cannot set the inference batchsize( default batchsize = 1 )
        /// </summary>
        /// <param name="imageList">image list</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public OcrBatchResult[] BatchAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            ValidationUtils.ValidationBatchPoolSize(Configuration.BatchPoolSize);
            return _executePipeline.BatchAsync(imageList, saveDir, processCallback, receiveAction);
        }
        /// <summary>
        /// Muti channel batch inference foreach api cannot set the inference batchsize( default batchsize = 1 )
        /// </summary>
        /// <param name="imageList">image list</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            ValidationUtils.ValidationBatchPoolSize(Configuration.BatchPoolSize);
            await foreach (var result in _executePipeline.BatchForeachAsync(imageList, saveDir, processCallback, receiveAction))
            {
                yield return result;
            }
        }
        /// <summary>
        /// Muti channel batch inference foreach api cannot set the inference batchsize( default batchsize = 1 )
        /// </summary>
        /// <param name="dir">images folder</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(string dir, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = ValidationUtils.ValidationImageBatch(dir);
            await foreach (var result in BatchForeachAsync(list, saveDir, processCallback, receiveAction))
            {
                yield return result;
            }
        }
        /// <summary>
        /// Muti channel batch inference can set the inference batchsize(ClassifierConfig.ClsBatchNum & RecognizerConfig.RecBatchNum), if the batch size is too large, out of memory will occur
        /// </summary>
        /// <param name="dir">images folder</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public OcrBatchResult[] BatchParallelAsync(string dir, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = ValidationUtils.ValidationImageBatch(dir);
            return BatchParallelAsync(list, saveDir, processCallback, receiveAction);
        }

        /// <summary>
        /// Muti channel batch inference can set the inference batchsize(ClassifierConfig.ClsBatchNum & RecognizerConfig.RecBatchNum), if the batch size is too large, out of memory will occur
        /// </summary>
        /// <param name="imageList">image list</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public OcrBatchResult[] BatchParallelAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            return _executePipeline.BatchParallelAsync(imageList, saveDir, processCallback, receiveAction);
        }

        /// <summary>
        /// Muti channel batch inference foreach api can set the inference batchsize(ClassifierConfig.ClsBatchNum & RecognizerConfig.RecBatchNum), if the batch size is too large, out of memory will occur
        /// </summary>
        /// <param name="dir">images folder</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public async IAsyncEnumerable<OcrBatchResult> BatchParallelForeachAsync(string dir, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = ValidationUtils.ValidationImageBatch(dir);
            await foreach (var result in BatchParallelForeachAsync(list, saveDir, processCallback, receiveAction))
            {
                yield return result;
            }
        }
        /// <summary>
        /// Muti channel batch inference foreach api can set the inference batchsize(ClassifierConfig.ClsBatchNum & RecognizerConfig.RecBatchNum), if the batch size is too large, out of memory will occur
        /// </summary>
        /// <param name="imageList">image list</param>
        /// <param name="saveDir">the save folder for draw result image(optional)</param>
        /// <param name="processCallback">callback interface(optional)</param>
        /// <param name="receiveAction">receive result action delegate(optional)</param>
        /// <returns>Result including detect result & classifier result(optional) recognize result & image path & text block  & performance time</returns>
        public async IAsyncEnumerable<OcrBatchResult> BatchParallelForeachAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            await foreach (var result in _executePipeline.BatchParallelForeachAsync(imageList, saveDir, processCallback, receiveAction))
            {
                yield return result;
            }
        }

        public void Dispose()
        {
            _executePipeline.Dispose();
        }
    }
}
