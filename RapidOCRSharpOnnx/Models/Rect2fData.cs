using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Models
{
    internal struct Rect2fData
    {
        public float XMin { get; set; }
        public float YMin { get; set; }
        public float XMax { get; set; }
        public float YMax { get; set; }

        public Rect2fData(float xMin, float yMin, float xMax, float yMax)
        {
            XMin = xMin;
            YMin = yMin;
            XMax = xMax;
            YMax = yMax;
        }
    }
}
