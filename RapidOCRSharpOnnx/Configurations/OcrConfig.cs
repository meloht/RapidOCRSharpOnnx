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
        public bool ReturnWordBox { get; set; } = true;
        public bool ReturnSingleCharBox { get; set; } = true;

        public DetectorConfig DetectorConfig { get; set; }

        public ClassifierConfig ClassifierConfig { get; set; }

        public RecognizerConfig RecognizerConfig { get; set; }

        public OcrConfig(string detectorModelPath, string recognizerModelPath, OCRVersion ocrVersion, string classifierModelPath = null)
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
            RecognizerConfig = new RecognizerConfig { ModelPath = recognizerModelPath };

            if (!string.IsNullOrWhiteSpace(classifierModelPath) || ocrVersion != OCRVersion.Null)
            {
                if (!string.IsNullOrWhiteSpace(classifierModelPath) && ocrVersion != OCRVersion.Null)
                {

                    ClassifierConfig = new ClassifierConfig { OCRVersion = ocrVersion, ModelPath = classifierModelPath };
                }
                else
                {
                    throw new ArgumentException("Classifier ModelPath is null or empty, or OCRVersion is Null.");
                }
            }
           
        }

        public OcrConfig(string detectorModelPath, string recognizerModelPath) : this(detectorModelPath, recognizerModelPath, OCRVersion.Null)
        {

        }



    }
}
