using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Utils
{
    internal class ValidationUtils
    {
        public static void ValidateImage(string imagePath)
        {
            if (string.IsNullOrEmpty(imagePath))
            {
                throw new ArgumentNullException(nameof(imagePath), $"{nameof(imagePath)} cannot be null or empty.");
            }
            if(!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"{nameof(imagePath)} does not exist.", imagePath);
            }
            string extension = Path.GetExtension(imagePath).ToLower();
            if (!IsImageByExtension(extension))
            {
                throw new ArgumentException($"{nameof(imagePath)} is not a valid image file.", nameof(imagePath));
            }
        }

       public static bool IsImageByExtension(string path)
        {
            string ext = Path.GetExtension(path).ToLower();
            return ext == ".jpg" || ext == ".jpeg" ||
                   ext == ".png" || ext == ".bmp" ||
                   ext == ".gif" || ext == ".tiff" ||
                   ext == ".webp";
        }
    }
}
