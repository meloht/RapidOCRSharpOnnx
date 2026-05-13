using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.TestCommon;
using RapidOCRSharpOnnx.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Test
{
    public class UnitTestApiNoCls : UnitTestBase, IDisposable
    {
        RapidOCRSharp _ocr;
        Dictionary<string, string> _dict;
        private string _dir;
        public UnitTestApiNoCls() : base()
        {
            _dir = GetImageHorizFolder();
            _dict = GetImagesHorizMap();
            _ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5)));
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
            var res = _ocr.BatchAsync(_dir);
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

            var res = _ocr.BatchAsync(_dir, processCallback: new BatchProcessCallback(_dict));
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
            var res = _ocr.BatchAsync(_dir, receiveAction: ReceiveResult);
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
            int idx = 0;
            await foreach (var item in _ocr.BatchForeachAsync(_dir))
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
            var res = _ocr.BatchParallelAsync(_dir);
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
            var res = _ocr.BatchParallelAsync(_dir, processCallback: new BatchProcessCallback(_dict));
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
            var res = _ocr.BatchParallelAsync(_dir, receiveAction: ReceiveResult);
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
            int idx = 0;
            await foreach (var item in _ocr.BatchParallelForeachAsync(_dir))
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
