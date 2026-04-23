using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx
{
    public interface IExecutePipeline : IDisposable
    {
        OcrResult RecognizeText(string imagePath, string savePath = null);
        OcrResult RecognizeText(Mat image, string savePath = null);

       // IAsyncEnumerable<OcrBatchResult> BatchForeachAsync(List<string> imageList);
        OcrBatchResult[] BatchAsync(List<string> imageList);
    }
}
