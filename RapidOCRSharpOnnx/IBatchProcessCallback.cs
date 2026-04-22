using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx
{
    public interface IBatchProcessCallback
    {
        void ReceiveProcessResult(OcrBatchResult result);
    }
}
