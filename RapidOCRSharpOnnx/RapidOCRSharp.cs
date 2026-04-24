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
        public OcrBatchResult[] BatchAsync(string dir, string saveDir = null)
        {
            var list = ValidationUtils.ValidationImageBatch(dir, Configuration.BatchPoolSize);
            return _executePipeline.BatchAsync(list, saveDir);
        }
        public OcrBatchResult[] BatchAsync(List<string> imageList, string saveDir = null)
        {
            var list = UtilsHelper.GetFilesFromListPaths(imageList);
            ValidationUtils.ValidationImageListCount(list);
            return _executePipeline.BatchAsync(imageList, saveDir);
        }

        //public async IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList)
        //{
        //    await foreach (var result in _executePipeline.BatchForeachAsync(imageList))
        //    {
        //        yield return result;
        //    }
        //}


        public void Dispose()
        {
            _executePipeline.Dispose();
        }
    }
}
