using OpenCvSharp;
using RadpidOCRCSharpOnnx.Config;
using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public class TextRecognizer
    {
        private OrtInferSession _session;
        public TextRecognizer()
        {
            _session = new OrtInferSession(RecConfig.ModelPath);
        }

        public void RecognizeTxt(Mat[] imgList)
        {
            
        }
    }
}
