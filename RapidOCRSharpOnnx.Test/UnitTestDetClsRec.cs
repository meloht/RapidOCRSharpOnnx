using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Test
{
    public class UnitTestDetClsRec: UnitTestBase, IDisposable
    {
        RapidOCRSharp _ocr;
        public UnitTestDetClsRec():base()
        {
            _ocr= new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath)));
        }

        public void Dispose()
        {
            _ocr.Dispose();
        }

        [Fact]
        public void Test01()
        {
            var res = _ocr.RecognizeText(GetFullPath(txtpng));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(txtRes, res.TextBlocks);
        }

        [Fact]
        public void Test02()
        {
            var res = _ocr.RecognizeText(GetFullPath(headerTextpng));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(txtRes, res.TextBlocks);
        }
    }
}
