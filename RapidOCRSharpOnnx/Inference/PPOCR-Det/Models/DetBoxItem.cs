using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Inference.PPOCR_Det.Models
{
    public class DetBoxItem
    {
        public Point2f[] Box { get; set; }
        public float Score { get; set; }
        public int LineId { get; set; }
        public string Word { get; set; }

        public DetBoxItem(Point2f[] box, float score, int lineId,string word)
        {
            Box = box;
            Score = score;
            LineId = lineId;
            Word = word;
        }

    }
}
