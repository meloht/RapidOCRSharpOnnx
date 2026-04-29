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

        public OcrResult RecognizeText(string imagePath, string savePath = null)
        {
            ValidationUtils.ValidateImage(imagePath);
            return _executePipeline.RecognizeText(imagePath, savePath);
        }
        public OcrResult RecognizeTextSeq(string imagePath, string savePath = null)
        {
            ValidationUtils.ValidateImage(imagePath);
            return _executePipeline.RecognizeTextSeq(imagePath, savePath);
        }
        public OcrResult RecognizeText(Mat image, string savePath = null)
        {
            return _executePipeline.RecognizeText(image, savePath);
        }
        public OcrBatchResult[] BatchAsync(string dir, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = ValidationUtils.ValidationImageBatch(dir, Configuration.BatchPoolSize);
            return _executePipeline.BatchAsync(list, saveDir, processCallback, receiveAction);
        }
        public OcrBatchResult[] BatchAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            return _executePipeline.BatchAsync(imageList, saveDir, processCallback, receiveAction);
        }

        public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            await foreach (var result in _executePipeline.BatchForeachAsync(imageList, saveDir, processCallback, receiveAction))
            {
                yield return result;
            }
        }
        public OcrBatchResult[] BatchParallelAsync(List<string> imageList, string saveDir = null, IBatchProcessCallback processCallback = null, Action<OcrBatchResult> receiveAction = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            return _executePipeline.BatchParallelAsync(imageList, saveDir, processCallback, receiveAction);
        }

        public void Dispose()
        {
            _executePipeline.Dispose();
        }
    }
}
