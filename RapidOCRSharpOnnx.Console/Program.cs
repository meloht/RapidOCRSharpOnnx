
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System.Diagnostics;
using System.Runtime.InteropServices;
namespace RapidOCRSharpOnnx.ConsoleApp
{
    internal class Program
    {
        const int _deviceId = 1;
        static void Main(string[] args)
        {
            var buildNumber = Environment.OSVersion.Version.Build;
            TestParallelBatch();
            //TestBatch();
            //_=TestBatchForeachAsync();
            //TestListSeq();
            // TestListSeq2();
            //TestImage();

            //Parallel.For(0, 100, i =>
            //{
            //    Console.WriteLine($"Index: {i}, Thread: {Thread.CurrentThread.ManagedThreadId}");
            //});
            Console.WriteLine("123");
            Console.ReadKey();

        }

        private static void TestImage()
        {
            string imgPath = @"E:\Hp\ai-image\ADFtools\headerText.png";
            //string imgPath = @"D:\code\model\OCRTestImages\text_vertical_words.png";
            //string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            //string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            //string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";

            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));

            string resPath = $"res_{Path.GetFileName(imgPath)}";
            var result = ocr.RecognizeText(imgPath, resPath);
            Console.WriteLine($"result: {result.ToString()}");
        }

        private static void TestListSeq()
        {
            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
            string saveDir = @"D:\code\model\OCRTestImagesResults";
            // string saveDir = null;
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
            Stopwatch sw = new Stopwatch();
            sw.Start();

            foreach (var item in list)
            {
                string resPath = Path.Combine(saveDir, $"res_{Path.GetFileName(item)}");
                var res = ocr.RecognizeText(item, resPath);
                Console.WriteLine(res);
            }

            sw.Stop();
            Console.WriteLine($"BatchAsync Time: {sw.ElapsedMilliseconds} ms");


            Console.WriteLine("end");
        }

        private static void TestListSeq2()
        {
            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
            //string saveDir = @"D:\code\model\OCRTestImagesResults";
            string saveDir = null;
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
            Stopwatch sw = new Stopwatch();
            sw.Start();

            foreach (var item in list)
            {
                var res = ocr.RecognizeTextSeq(item);
                Console.WriteLine(res);
            }

            sw.Stop();
            Console.WriteLine($"BatchAsync Time: {sw.ElapsedMilliseconds} ms");


            Console.WriteLine("end");
        }

        private static void TestBatch()
        {

            //string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            //string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            //string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";
            string saveDir = @"D:\code\model\OCRTestImagesResults";
            //string saveDir = null;
            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            var resPath = ocr.BatchAsync(list.ToList(), saveDir, receiveAction: ReceiveResult);
            sw.Stop();
            Console.WriteLine($"BatchAsync Time: {sw.ElapsedMilliseconds} ms");


            Console.WriteLine("end");
        }

        private static void TestParallelBatch()
        {

            //string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            //string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            //string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";
            string saveDir = @"D:\code\model\OCRTestImagesResults";
            //string saveDir = null;
            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            var resPath = ocr.BatchParallelAsync(list.ToList(), saveDir, receiveAction: ReceiveResult);
            sw.Stop();
            Console.WriteLine($"BatchAsync Time: {sw.ElapsedMilliseconds} ms");


            Console.WriteLine("end");
        }

        private static void ReceiveResult(OcrBatchResult batchResult)
        {
            Console.WriteLine(batchResult.ToString());
            Console.WriteLine("------------------------------------------------------------");
        }



        private static async Task TestBatchForeachAsync()
        {

            //string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            //string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            //string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";

            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
            var resPaths = ocr.BatchForeachAsync(list.ToList(), @"D:\code\model\OCRTestImagesResults");

            await foreach (var item in resPaths)
            {
                Console.WriteLine(item.TextBlocks);
            }

            Console.WriteLine("end");
        }
    }
}
