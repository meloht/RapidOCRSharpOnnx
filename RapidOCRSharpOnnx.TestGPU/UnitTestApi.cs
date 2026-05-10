using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.TestCommon;
using RapidOCRSharpOnnx.Utils;
using System.Management;

namespace RapidOCRSharpOnnx.TestGPU
{
    public class UnitTestApi: UnitTestBase,IDisposable
    {
        RapidOCRSharp _ocr;
        Dictionary<string, string> _dict;
        private int _deviceId;
        public UnitTestApi() : base()
        {
            _dict = GetImagesMap();
            _deviceId = Utils.GetMainGPU();
            _ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath), _deviceId));
        }

        public void Dispose()
        {
            _ocr.Dispose();
        }

        [Fact]
        public void TestRecognizeTextSeq()
        {
            var res = _ocr.RecognizeTextSeq(GetFullPath(png_txt));
            Assert.NotNull(res.TextBlocks);
            Assert.Equal(Res_txt, res.TextBlocks);
        }

        [Fact]
        public void TestBatchAsync()
        {
            string dir = GetImageFolder();

            var res = _ocr.BatchAsync(dir);
            Assert.NotNull(res);
            Assert.Equal(_dict.Count, res.Length);

            foreach (var item in res)
            {
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

        }

        [Fact]
        public void TestBatchCallbackAsync()
        {
            string dir = GetImageFolder();

            var res = _ocr.BatchAsync(dir, processCallback: new BatchProcessCallback(_dict));
            Assert.NotNull(res);
            Assert.Equal(_dict.Count, res.Length);

            foreach (var item in res)
            {
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

        }

        [Fact]
        public void TestBatchActionAsync()
        {
            string dir = GetImageFolder();

            var res = _ocr.BatchAsync(dir, receiveAction: ReceiveResult);
            Assert.NotNull(res);
            Assert.Equal(_dict.Count, res.Length);

            foreach (var item in res)
            {
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

        }


        [Fact]
        public async Task TestBatchForeachAsync()
        {
            string dir = GetImageFolder();
            int idx = 0;
            await foreach (var item in _ocr.BatchForeachAsync(dir))
            {
                idx++;
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

            Assert.Equal(_dict.Count, idx);
        }




        [Fact]
        public void TestBatchParallelAsync()
        {
            string dir = GetImageFolder();

            var res = _ocr.BatchParallelAsync(dir);
            Assert.NotNull(res);
            Assert.Equal(_dict.Count, res.Length);

            foreach (var item in res)
            {
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

        }
        [Fact]
        public void TestBatchParallelCallbackAsync()
        {
            string dir = GetImageFolder();

            var res = _ocr.BatchParallelAsync(dir, processCallback: new BatchProcessCallback(_dict));
            Assert.NotNull(res);
            Assert.Equal(_dict.Count, res.Length);

            foreach (var item in res)
            {
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

        }

        [Fact]
        public void TestBatchParallelActionAsync()
        {
            string dir = GetImageFolder();

            var res = _ocr.BatchParallelAsync(dir, receiveAction: ReceiveResult);
            Assert.NotNull(res);
            Assert.Equal(_dict.Count, res.Length);

            foreach (var item in res)
            {
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }
        }


        [Fact]
        public async Task TestBatchParallelForeachAsync()
        {
            string dir = GetImageFolder();

            int idx = 0;
            await foreach (var item in _ocr.BatchParallelForeachAsync(dir))
            {
                idx++;
                string name = Path.GetFileName(item.ImagePath);
                Assert.Equal(_dict[name], item.TextBlocks);
            }

            Assert.Equal(_dict.Count, idx);

        }




        private void ReceiveResult(OcrBatchResult result)
        {
            string name = Path.GetFileName(result.ImagePath);
            Assert.Equal(_dict[name], result.TextBlocks);
        }


        class BatchProcessCallback : IBatchProcessCallback
        {
            Dictionary<string, string> _dict;
            public BatchProcessCallback(Dictionary<string, string> dict)
            {
                _dict = dict;
            }
            public void ReceiveProcessResult(OcrBatchResult result)
            {
                string name = Path.GetFileName(result.ImagePath);
                Assert.Equal(_dict[name], result.TextBlocks);
            }
        }
    }
}
