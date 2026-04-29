using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System.Diagnostics;

namespace RapidOCRSharpOnnx.ConsoleGPU
{
    internal class Program
    {
        const int _deviceId = 0;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            TestParallelBatch();
        }


        private static void TestListSeq()
        {
            string detectPath = @"C:\deeplearning\gitCode\meloht\RapidOCRSharpOnnx\RapidOCRSharpOnnx.Test\Models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"C:\deeplearning\gitCode\meloht\RapidOCRSharpOnnx\RapidOCRSharpOnnx.Test\Models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"C:\deeplearning\gitCode\meloht\RapidOCRSharpOnnx\RapidOCRSharpOnnx.Test\Models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
            //string saveDir = @"C:\code\model\OCRTestImagesResults";
            string saveDir = null;
            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCUDA(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"C:\code\model\OCRTestImages");
            Stopwatch sw = new Stopwatch();
            sw.Start();

            foreach (var item in list)
            {
               // string resPath = Path.Combine(saveDir, $"res_{Path.GetFileName(item)}");
                var res = ocr.RecognizeText(item, null);
                Console.WriteLine(res);
            }

            sw.Stop();
            Console.WriteLine($"BatchAsync Time: {sw.ElapsedMilliseconds} ms");


            Console.WriteLine("end");
        }
        private static void TestParallelBatch()
        {

            //string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            //string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            //string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";
          
            //string saveDir = null;
            string detectPath = @"C:\deeplearning\gitCode\meloht\RapidOCRSharpOnnx\RapidOCRSharpOnnx.Test\Models\ch_PP-OCRv5_det_mobile.onnx";
            string recogPath = @"C:\deeplearning\gitCode\meloht\RapidOCRSharpOnnx\RapidOCRSharpOnnx.Test\Models\ch_PP-OCRv5_rec_mobile.onnx";
            string clsPath = @"C:\deeplearning\gitCode\meloht\RapidOCRSharpOnnx\RapidOCRSharpOnnx.Test\Models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
            string saveDir = @"C:\code\model\OCRTestImagesResults";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCUDA(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"C:\code\model\OCRTestImages");
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
    }
}
