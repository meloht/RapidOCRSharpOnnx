using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.InferenceEngine
{
    public struct WordInfo
    {
        public List<List<char>> Words { get; set; }

        public List<List<int>> WordCols { get; set; }

        public List<WordType> WordTypes { get; set; }

        public WordInfo(List<List<char>> words, List<List<int>> wordCols, List<WordType> wordTypes)
        {
            Words = words;
            WordCols = wordCols;
            WordTypes = wordTypes;
        }
    }
}
