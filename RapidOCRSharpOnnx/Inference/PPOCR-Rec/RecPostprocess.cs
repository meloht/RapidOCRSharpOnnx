using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Models;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class RecPostprocess : IRecPostprocess
    {
        private OcrConfig _ocrConfig;
        public RecPostprocess(OcrConfig ocrConfig)
        {
            _ocrConfig = ocrConfig;
        }
        public RecResult[] RecPostProcess(OrtValue ortValue, float[] wh_ratio_list, float max_wh_ratio, string[] charList)
        {
            var shapeInfo = ortValue.GetTensorTypeAndShape();
            int batchSize = (int)shapeInfo.Shape[0];
            int tNum = (int)shapeInfo.Shape[1];
            int numClasses = (int)shapeInfo.Shape[2];

            var data = ortValue.GetTensorDataAsSpan<float>();

            var maxIndexAndValue = GetMaxIndexAndValue(data, batchSize, tNum, numClasses);
            int[] ignored_tokens = getIgnoredTokens();

            RecResult[] results = new RecResult[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                int[] token_indices = maxIndexAndValue.Indices[i];
                bool[] selection = GetSelection(token_indices, ignored_tokens);

                var confList = GetConfList(maxIndexAndValue.Values, i, selection);

                string text = GetCharList(token_indices, selection, charList);
                float avgConf = (float)Math.Round(confList.Average(), 5);

                results[i] = new RecResult(text, avgConf);
                results[i].LineTxtLen = token_indices.Length * wh_ratio_list[i] / max_wh_ratio;
                results[i].ValidCols = GetValidCols(selection);
                results[i].ConfList = confList;
            }
            return results;

        }

        private ValueTuplePairArray GetMaxIndexAndValue(ReadOnlySpan<float> data, int batchSize, int tNum, int numClasses)
        {
            float[][] maxValues = new float[batchSize][];
            int[][] maxIndices = new int[batchSize][];
            int maxIdx = 0;
            float maxVal = float.MinValue;
            int idx = 0;
            for (int i = 0; i < batchSize; i++)
            {
                maxValues[i] = new float[tNum];
                maxIndices[i] = new int[tNum];

                for (int j = 0; j < tNum; j++)
                {
                    maxIdx = 0;
                    maxVal = float.MinValue;

                    for (int k = 0; k < numClasses; k++)
                    {
                        float val = data[idx++];
                        if (val > maxVal)
                        {
                            maxVal = val;
                            maxIdx = k;
                        }
                    }

                    maxValues[i][j] = maxVal;
                    maxIndices[i][j] = maxIdx;
                }
            }

            return new ValueTuplePairArray(maxIndices, maxValues);

        }

        private List<int> GetValidCols(bool[] selection)
        {
            var validCol = new List<int>();
            for (int i = 0; i < selection.Length; i++)
            {
                if (selection[i])
                    validCol.Add(i);
            }
            return validCol;
        }
   
        private List<float> GetConfList(float[][] values, int batchIdx, bool[] selection)
        {
            // 获取置信度列表
            List<float> confList;
            if (values != null)
            {
                confList = new List<float>();
                for (int i = 0; i < values[batchIdx].Length; i++)
                {
                    if (selection[i])
                        confList.Add(values[batchIdx][i]);
                }
                // 四舍五入到5位小数
                confList = confList.Select(c => (float)Math.Round(c, 5)).ToList();
            }
            else
            {
                float[] arr = new float[selection.Length];
                Array.Fill(arr, 1f);
                confList = arr.ToList();
            }

            if (confList.Count == 0)
                confList = [0f];

            return confList;
        }

        private string GetCharList(int[] tokenIndices, bool[] selection, string[] charList)
        {
            StringBuilder txt = new StringBuilder();

            for (int i = 0; i < tokenIndices.Length; i++)
            {
                if (selection[i])
                {
                    txt.Append(charList[tokenIndices[i]]);
                }
            }

            return txt.ToString();
        }

        private bool[] GetSelection(int[] token_indices, int[] ignored_tokens)
        {
            bool[] selection = new bool[token_indices.Length];
            Array.Fill(selection, true);

            for (int ii = 1; ii < token_indices.Length; ii++)
            {
                selection[ii] = token_indices[ii] != token_indices[ii - 1];
            }

            foreach (int ignored in ignored_tokens)
            {
                for (int i = 0; i < token_indices.Length; i++)
                {
                    selection[i] &= (token_indices[i] != ignored);
                }
            }

            return selection;
        }

        private int[] getIgnoredTokens()
        {
            return [0];//for ctc blank
        }
    }
}
