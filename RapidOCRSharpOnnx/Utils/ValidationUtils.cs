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
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"{nameof(imagePath)} does not exist.", imagePath);
            }
            string extension = Path.GetExtension(imagePath).ToLower();
            if (!UtilsHelper.IsImageByExtension(extension))
            {
                throw new ArgumentException($"{nameof(imagePath)} is not a valid image file.", nameof(imagePath));
            }
        }

        public static void ValidationImageListCount(List<string> list)
        {

            if (list == null || list.Count == 0)
            {
                string[] exts = UtilsHelper.GetImageExtensions();
                throw new ArgumentNullException($"list is null or empty, and there no any images in the list for image ext({string.Join(',', exts)})");
            }

        }



        public static List<string> ValidationImageBatch(string imgDir, int batchSize)
        {
            if (string.IsNullOrWhiteSpace(imgDir))
            {
                throw new ArgumentNullException($"imgDir is null or empty");
            }
            if (!Directory.Exists(imgDir))
            {
                throw new DirectoryNotFoundException($"{imgDir} directory not found");
            }
            if (batchSize <= 0)
            {
                throw new ArgumentNullException("batchSize must be greater than zero");
            }

            var files = UtilsHelper.GetFilesFromDirectory(imgDir);
            if (files.Count == 0)
            {
                string[] exts = UtilsHelper.GetImageExtensions();
                throw new ArgumentNullException($"there no any images in the directory for image ext({string.Join(',', exts)})");
            }
            return files;
        }
    }
}
