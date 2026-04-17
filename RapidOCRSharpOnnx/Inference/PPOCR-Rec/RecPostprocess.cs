using Microsoft.ML.OnnxRuntime;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec
{
    public class RecPostprocess: IRecPostprocess
    {
        private OcrConfig _ocrConfig;
        public RecPostprocess(OcrConfig ocrConfig)
        {
            _ocrConfig = ocrConfig;
        }
        public InferenceResult[] RecPostProcess(OrtValue ortValue, float[] wh_ratio_list, float max_wh_ratio, string[] charList)
        {
            var shapeInfo = ortValue.GetTensorTypeAndShape();
            int batchSize = (int)shapeInfo.Shape[0];
            int tNum = (int)shapeInfo.Shape[1];
            int numClasses = (int)shapeInfo.Shape[2];

            var data = ortValue.GetTensorDataAsSpan<float>();

            var maxIndexAndValue = GetMaxIndexAndValue(data, batchSize, tNum, numClasses);
            int[] ignored_tokens = getIgnoredTokens();

            InferenceResult[] results = new InferenceResult[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                int[] token_indices = maxIndexAndValue.Indices[i];
                bool[] selection = GetSelection(token_indices, ignored_tokens);

                var confList = GetConfList(maxIndexAndValue.Values, i, selection);

                string text = GetCharList(token_indices, selection, charList);
                float avgConf = (float)Math.Round(confList.Average(), 5);

                results[i] = new InferenceResult(text, avgConf);
                if (_ocrConfig.ReturnWordBox)
                {
                    var wordInfo = GetWordInfo(text, selection, confList);
                    // 这里可以根据 wordInfo 进行进一步处理，例如返回单词边界框等
                    wordInfo.LineTxtLen = token_indices.Length * wh_ratio_list[i] / max_wh_ratio;

                    results[i].WordInfo = wordInfo;
                }
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

        private WordInfo GetWordInfo(string text, bool[] selection, List<float> confList)
        {
            var validCol = new List<int>();
            for (int i = 0; i < selection.Length; i++)
            {
                if (selection[i])
                    validCol.Add(i);
            }
            if (validCol.Count == 0)
                return new WordInfo(); // 无有效字符

            float[] colWidth = new float[validCol.Count];
            for (int i = 1; i < validCol.Count; i++)
            {
                colWidth[i] = validCol[i] - validCol[i - 1];
            }

            int firstColValue = validCol[0];
            int minVal = text.Length > 0 && UtilsHelper.IsChineseChar(text[0]) ? 3 : 2;
            colWidth[0] = Math.Min(minVal, firstColValue);

            var wordList = new List<char[]>();
            var wordColList = new List<int[]>();
            var stateList = new List<WordType>();

            var wordConfList = new List<float[]>();

            var wordContent = new List<char>();
            var wordColContent = new List<int>();
            var confArr = new List<float>();


            WordType? state = null;

            for (int cIdx = 0; cIdx < text.Length; cIdx++)
            {
                char ch = text[cIdx];

                // 处理空白字符：结束当前单词
                if (char.IsWhiteSpace(ch))
                {
                    if (wordContent.Count > 0)
                    {
                        wordList.Add(wordContent.ToArray());
                        wordColList.Add(wordColContent.ToArray());
                        wordConfList.Add(confArr.ToArray());
                        stateList.Add(state!.Value);
                        wordContent.Clear();
                        wordColContent.Clear();
                        confArr.Clear();
                    }

                    continue;
                }


                // 判断当前字符类型
                WordType cState = UtilsHelper.IsChineseChar(ch) ? WordType.CN : WordType.EN_NUM;
                if (state == null)
                    state = cState;

                // 类型变化或列宽过大（>5）时切分单词
                if (state != cState || colWidth[cIdx] > 5)
                {
                    if (wordContent.Count > 0)
                    {
                        wordList.Add(wordContent.ToArray());
                        wordColList.Add(wordColContent.ToArray());
                        wordConfList.Add(confArr.ToArray());
                        stateList.Add(state.Value);
                        wordContent.Clear();
                        wordColContent.Clear();
                        confArr.Clear();

                    }
                    state = cState;
                }

                // 将当前字符加入正在构建的单词
                wordContent.Add(ch);
                wordColContent.Add(validCol[cIdx]);
                confArr.Add(confList[cIdx]);
            }

            // 处理最后一个单词
            if (wordContent.Count > 0)
            {
                wordList.Add(wordContent.ToArray());
                wordColList.Add(wordColContent.ToArray());
                stateList.Add(state!.Value);
                wordConfList.Add(confArr.ToArray());
            }

            return new WordInfo(wordList, wordColList, stateList, wordConfList);

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

            Console.WriteLine("ConfList: " + string.Join(", ", confList));
            return confList;
        }

        private string GetCharList(int[] tokenIndices, bool[] selection, string[] charList)
        {
            StringBuilder txt = new StringBuilder();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < tokenIndices.Length; i++)
            {
                if (selection[i])
                {
                    txt.Append(charList[tokenIndices[i]]);
                    sb.Append(tokenIndices[i]).Append(",");
                }
            }
            Console.WriteLine(txt.ToString());
            Console.WriteLine(sb.ToString());
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
