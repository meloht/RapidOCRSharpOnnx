using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class OcrConfig
    {
        public float DrawHorizontalFontScaling { get; set; } = 0.86f;
        public float DrawVerticalFontScaling { get; set; } = 0.62f;
        public int MinHeight { get; set; } = 30;
        public int WidthHeightRatio { get; set; } = 8;
        public int MaxSideLen { get; set; } = 2000;
        public int MinSideLen { get; set; } = 30;
        public bool ReturnWordBox { get; set; } = false;
        public bool ReturnSingleCharBox { get; set; } = false;
        private int _batchPoolSize = 30;

        public string FontPath { get; set; }
        public LangRec LangRec { get; set; }

        public int BatchPoolSize
        {
            get { return _batchPoolSize; }
            set
            {
                if (value < 1 && value > 100)
                {
                    throw new ArgumentException("The BatchPoolSize must be between 1 and 100");
                }
                _batchPoolSize = value;
            }
        }

        public DetectorConfig DetectorConfig { get; set; }

        public ClassifierConfig ClassifierConfig { get; set; }

        public RecognizerConfig RecognizerConfig { get; set; }

        /// <summary>
        /// OCR config
        /// </summary>
        /// <param name="detectorModelPath">text detection model path</param>
        /// <param name="recognizerModelPath">text recognition model path</param>
        /// <param name="langFont">draw result image font</param>
        /// <param name="ocrVersion">PP-OCR version</param>
        /// <param name="classifierModelPath">text line orientation classification model path</param>
        /// <exception cref="ArgumentException">Detector ModelPath is null or empty. or Recognizer ModelPath is null or empty.</exception>
        public OcrConfig(string detectorModelPath, string recognizerModelPath, LangRec langFont, OCRVersion ocrVersion, string classifierModelPath = null)
        {
            if (string.IsNullOrWhiteSpace(detectorModelPath))
            {
                throw new ArgumentException("Detector ModelPath is null or empty.");
            }
            if (string.IsNullOrWhiteSpace(recognizerModelPath))
            {
                throw new ArgumentException("Recognizer ModelPath is null or empty.");
            }
            LangRec = langFont;
            DetectorConfig = new DetectorConfig { ModelPath = detectorModelPath, OCRVersion = ocrVersion };
            RecognizerConfig = new RecognizerConfig { ModelPath = recognizerModelPath };

            if (!string.IsNullOrWhiteSpace(classifierModelPath))
            {
                ClassifierConfig = new ClassifierConfig { OCRVersion = ocrVersion, ModelPath = classifierModelPath };
            }

        }
        /// <summary>
        /// OCR config
        /// </summary>
        /// <param name="detectorModelPath">text detection model path</param>
        /// <param name="recognizerModelPath">text recognition model path</param>
        /// <param name="langRec">draw result image font</param>
        /// <param name="ocrVersion">PP-OCR version</param>
        public OcrConfig(string detectorModelPath, string recognizerModelPath, LangRec langRec, OCRVersion ocrVersion) : this(detectorModelPath, recognizerModelPath, langRec, ocrVersion, null)
        {

        }
        /// <summary>
        /// OCR config
        /// </summary>
        /// <param name="detectorModelPath">text detection model path</param>
        /// <param name="recognizerModelPath">text recognition model path</param>
        /// <param name="fontPath">draw result image font file path</param>
        /// <param name="ocrVersion">PP-OCR version</param>
        /// <param name="classifierModelPath">text line orientation classification model path</param>
        /// <exception cref="ArgumentException">fontPath is not exist</exception>
        public OcrConfig(string detectorModelPath, string recognizerModelPath, string fontPath, OCRVersion ocrVersion, string classifierModelPath = null)
            : this(detectorModelPath, recognizerModelPath, LangRec.CH, ocrVersion, classifierModelPath)
        {
            if (!File.Exists(fontPath))
            {
                throw new ArgumentException($"{fontPath} fontPath is not exist");
            }
            FontPath = fontPath;
        }



    }
}
