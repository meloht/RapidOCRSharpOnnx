using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Utils
{
    public class ImageBatchData : IDisposable
    {
        public Mat ResizedImg { get; set; }
        public FixedBuffer FixedBuffer { get; set; }

        public OrtValue InputOrtValue { get; set; }

        public ImageBatchData(int inputSizeInBytes, long[] inputShape)
        {
            ResizedImg = new Mat();
            FixedBuffer = new FixedBuffer(inputSizeInBytes);
            InputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float, inputShape, FixedBuffer.Address, inputSizeInBytes);
        }

        public void Dispose()
        {
            ResizedImg?.Dispose();
            FixedBuffer?.Dispose();
            InputOrtValue?.Dispose();
        }
    }
}
