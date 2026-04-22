using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Configurations
{
    public class OcrConfig
    {
        public float TextScore { get; set; } = 0.5f;
        public int MinHeight { get; set; } = 30;
        public int WidthHeightRatio { get; set; } = 8;
        public int MaxSideLen { get; set; } = 2000;
        public int MinSideLen { get; set; } = 30;
        public bool ReturnWordBox { get; set; } = false;
        public bool ReturnSingleCharBox { get; set; } = false;
        private int _batchPoolSize = 30;
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

        public OcrConfig(string detectorModelPath, string recognizerModelPath, LangRec langRec, OCRVersion ocrVersion, string classifierModelPath = null)
        {
            if (string.IsNullOrWhiteSpace(detectorModelPath))
            {
                throw new ArgumentException("Detector ModelPath is null or empty.");
            }
            if (string.IsNullOrWhiteSpace(recognizerModelPath))
            {
                throw new ArgumentException("Recognizer ModelPath is null or empty.");
            }

            DetectorConfig = new DetectorConfig { ModelPath = detectorModelPath };
            RecognizerConfig = new RecognizerConfig { ModelPath = recognizerModelPath, LangRec = langRec };

            if (!string.IsNullOrWhiteSpace(classifierModelPath) )
            {
                ClassifierConfig = new ClassifierConfig { OCRVersion = ocrVersion, ModelPath = classifierModelPath };
            }

        }

        public OcrConfig(string detectorModelPath, string recognizerModelPath, LangRec langRec, OCRVersion ocrVersion) : this(detectorModelPath, recognizerModelPath, langRec, ocrVersion, null)
        {

        }



    }
}
