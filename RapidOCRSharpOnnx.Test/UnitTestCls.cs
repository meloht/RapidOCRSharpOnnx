using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;

namespace RapidOCRSharpOnnx.Test
{
    public class UnitTestCls: UnitTestBase
    {
        [Fact]
        public void TestMobile()
        {
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath)));
            var res = ocr.RecognizeText(GetFullPath(testClspng));
            Assert.NotNull(res.ClsResult.Data);
            Assert.True(res.ClsResult.Data.Length > 0);
            Assert.Equal("180", res.ClsResult.Data[0].Label);
        }

        [Fact]
        public void TestServer()
        {
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsServerPath)));
            var res = ocr.RecognizeText(GetFullPath(testClspng));
            Assert.NotNull(res.ClsResult.Data);
            Assert.True(res.ClsResult.Data.Length > 0);
            Assert.Equal("180", res.ClsResult.Data[0].Label);
        }
    }
}
