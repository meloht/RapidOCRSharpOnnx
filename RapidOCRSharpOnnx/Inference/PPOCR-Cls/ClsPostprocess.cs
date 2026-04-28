using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Cls
{
    public class ClsPostprocess: IClsPostprocess
    {
        private ClassifierConfig _classifierConfig;
        public ClsPostprocess(ClassifierConfig clsConfig)
        {
            _classifierConfig = clsConfig;
        }

        public ClsResult ClsPostProcess(OrtValue ortValue, Mat img)
        {
            var shapeInfo = ortValue.GetTensorTypeAndShape();
    
            int numClasses = (int)shapeInfo.Shape[1];
           
            var data = ortValue.GetTensorDataAsSpan<float>();
            if (data.Length != numClasses)
                throw new InvalidOperationException("Data length mismatch.");

            int maxIdx = 0;
            float maxVal = float.MinValue;

            for (int j = 0; j < numClasses; j++)
            {
                float val = data[j];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = j;
                }
            }

            string label = _classifierConfig.LabelList[maxIdx];
            float score = maxVal;
            
         
            if (label == "180" && score > _classifierConfig.ClsThresh)
            {
                Cv2.Rotate(img, img, RotateFlags.Rotate180);
            }
            return new ClsResult(label, score);
        }
       
        public void ClsPostProcess(OrtValue ortValue,int batchIndex, DisposableList<ImageIndex> imgList, ClsResult[] cls_res)
        {
            var shapeInfo = ortValue.GetTensorTypeAndShape();
            int batchSize = (int)shapeInfo.Shape[0];
            int numClasses = (int)shapeInfo.Shape[1];

            var data = ortValue.GetTensorDataAsSpan<float>();
            if (data.Length != batchSize * numClasses)
                throw new InvalidOperationException("Cls Data length mismatch.");

            int idx = 0;
            int maxIdx = 0;
            float maxVal = float.MinValue;
            for (int i = 0; i < batchSize; i++)
            {
                maxIdx = 0;
                maxVal = float.MinValue;

                for (int j = 0; j < numClasses; j++)
                {
                    float val = data[idx++];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
     
                string label = _classifierConfig.LabelList[maxIdx];
                float score = maxVal;
                int index= batchIndex + i;
                cls_res[index].Label = label;
                cls_res[index].Score = score;
                if (label == "180" && score > _classifierConfig.ClsThresh)
                {
                    Cv2.Rotate(imgList[index].Image, imgList[index].Image, RotateFlags.Rotate180);
                }
            }

        }
    }
}
