
using OpenCvSharp;
using RapidOCRSharpOnnx.Configurations;
using RapidOCRSharpOnnx.Providers;
using RapidOCRSharpOnnx.Utils;
using System.Runtime.InteropServices;
namespace RapidOCRSharpOnnx.ConsoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var buildNumber = Environment.OSVersion.Version.Build;
            //var dd= NativeMethods.core_getVersionRevision
            Console.WriteLine("Hello, World!");
            string imgPath = @"D:\code\RapidOCR-3.8.0\python\tests\test_files\latin.jpg";
            string detectPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_det_mobile.onnx";
            string recogPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_PP-OCRv4_rec_mobile.onnx";
            string clsPath = @"D:\code\RapidOCR-3.8.0\python\rapidocr\models\ch_ppocr_mobile_v2.0_cls_mobile.onnx";

            using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recogPath, LangRec.EN, OCRVersion.PPOCRV4, clsPath)));

            string resPath = $"res_{Path.GetFileName(imgPath)}";
            var result = ocr.RecognizeText(imgPath, resPath);
            Console.WriteLine($"result: {result}");

        }
    }
}
