using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Rec.Models
{
    public class ImageIndex : IDisposable
    {
        public Mat Image { get; } 
        public int Index { get; } 
        public ImageIndex(Mat image, int index)
        {
            Image = image;
            Index = index;
        }
        public void Dispose()
        {
            Image?.Dispose();
        }
    }

}
