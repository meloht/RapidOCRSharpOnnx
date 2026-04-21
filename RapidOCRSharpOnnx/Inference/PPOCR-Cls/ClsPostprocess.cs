using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
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
        public void ClsPostProcess(OrtValue ortValue,int ij, Mat[] imgList, ClsResult[] cls_res)
        {
            var shapeInfo = ortValue.GetTensorTypeAndShape();
            int batchSize = (int)shapeInfo.Shape[0];
            int numClasses = (int)shapeInfo.Shape[1];

            var data = ortValue.GetTensorDataAsSpan<float>();
            if (data.Length != batchSize * numClasses)
                throw new InvalidOperationException("Data length mismatch.");

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
                int index= ij + i;
                cls_res[index].Label = label;
                cls_res[index].Score = score;
                if (label == "180" && score > _classifierConfig.ClsThresh)
                {
                    Cv2.Rotate(imgList[index], imgList[index], RotateFlags.Rotate180);
                }
            }

        }
    }
}
