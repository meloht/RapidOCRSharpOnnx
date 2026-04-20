using OpenCvSharp;
using OpenCvSharp.ML;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference;
using RapidOCRSharpOnnx.Models;
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

        public OcrConfig OcrConfig
        {
            get { return _executionProvider.OcrConfig; }
        }


        public RapidOCRSharp(IExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
            _ocrDetector = _executionProvider.CreateDetector();
            _ocrClassifier = _executionProvider.CreateClassifier();
            _ocrRecognizer = _executionProvider.CreateRecognizer();
            _ocrDrawerSkia = new OcrDrawerSkia(OcrConfig);
        }

        public void RecognizeText(string imagePath, string savePath = null)
        {
            using Mat image = Cv2.ImRead(imagePath);
            RecognizeText(image, savePath);
        }
        public void RecognizeText(Mat image, string savePath = null)
        {
            var detResult = _ocrDetector.TextDetect(image);
            if (_ocrClassifier != null)
            {
                var clsBoxes = _ocrClassifier.TextClassify(detResult.ImgCropList);
            }

            var recResults = _ocrRecognizer.TextRecognize(detResult.ImgCropList);

            if (!string.IsNullOrEmpty(savePath))
            {
                _ocrDrawerSkia.DrawTextBlock(image, savePath, detResult, recResults);
            }
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
