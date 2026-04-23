
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System.Runtime.InteropServices;
namespace RapidOCRSharpOnnx.ConsoleApp
{
    internal class Program
    {
        const int _deviceId = 1;
        static void Main(string[] args)
        {
            var buildNumber = Environment.OSVersion.Version.Build;
            TestBatch();
            //TestImage();
        }

        private static void TestImage()
        {
            string imgPath = @"D:\code\model\OCRTestImages\blank.png";
            //string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_det_mobile.onnx";
            //string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv5_rec_mobile.onnx";
            //string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";

            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.EN, OCRVersion.PPOCRV5, clsPath), _deviceId));

            string resPath = $"res_{Path.GetFileName(imgPath)}";
            var result = ocr.RecognizeText(imgPath, resPath);
            Console.WriteLine($"result: {result.ToString()}");
        }

        private static void TestBatch()
        {

            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.EN, OCRVersion.PPOCRV5, clsPath), _deviceId));
            var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
            var resPath = ocr.BatchAsync(list.ToList());

            foreach (var item in resPath)
            {
                Console.WriteLine(item.TextBlocks);
            }

            Console.WriteLine("end");
        }
    }
}
