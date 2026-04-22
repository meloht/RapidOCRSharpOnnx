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
        private IOcrDetector _ocrDetector;
        private IOcrClassifier _ocrClassifier;
        private IOcrRecognizer _ocrRecognizer;
        private OcrDrawerSkia _ocrDrawerSkia;

        public OcrConfig Configuration
        {
            get { return _executionProvider.OcrConfig; }
        }


        public RapidOCRSharp(IExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
            _ocrDetector = _executionProvider.CreateDetector();
            _ocrClassifier = _executionProvider.CreateClassifier();
            _ocrRecognizer = _executionProvider.CreateRecognizer();
            _ocrDrawerSkia = new OcrDrawerSkia(Configuration);
        }

        public OcrResult RecognizeText(string imagePath, string savePath = null)
        {
            ValidationUtils.ValidateImage(imagePath);
            using Mat image = Cv2.ImRead(imagePath);
            return RecognizeText(image, savePath);
        }
        public OcrResult RecognizeText(Mat image, string savePath = null)
        {
            OcrResult result = new OcrResult();
            var detResult = _ocrDetector.TextDetect(image);
            result.DetResults = detResult;
            using (detResult.Data.ImgCropList)
            {
                if (_ocrClassifier != null)
                {
                    var ClsResult = _ocrClassifier.TextClassify(detResult.Data.ImgCropList);
                    result.ClsResults = ClsResult;
                }

                var recResults = _ocrRecognizer.TextRecognize(detResult.Data.ImgCropList);
                result.RecResults = recResults;

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
