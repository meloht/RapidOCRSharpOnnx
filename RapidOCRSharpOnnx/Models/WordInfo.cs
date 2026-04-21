using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Models
{
    public struct WordInfo
    {
        public List<char[]> Words { get; set; }

        public List<int[]> WordCols { get; set; }

        public List<WordType> WordTypes { get; set; }

        public float LineTxtLen { get; set; }
        public List<float[]> Confs { get; set; }

        public WordInfo(List<char[]> words, List<int[]> wordCols, List<WordType> wordTypes, List<float[]> confs)
        {
            Words = words;
            WordCols = wordCols;
            WordTypes = wordTypes;
            Confs = confs;
        }
    }
}
