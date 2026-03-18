using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RadpidOCRCSharpOnnx.Utils
{
    public static class UtilsHelper
    {
        public static void MakeDir(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
        }

        public static bool IsChineseChar(char ch)
        {
            // 对应Python的Unicode范围判断：
            // \u4e00-\u9fff：汉字
            // \u3000-\u303f：CJK标点（。、“”《》等）
            // \uff00-\uffef：全角符号（，．！？【】等）
            return (ch >= '\u4e00' && ch <= '\u9fff')
                   || (ch >= '\u3000' && ch <= '\u303f')
                   || (ch >= '\uff00' && ch <= '\uffef');
        }

        public static bool HasChineseChar(string text)
        {
            // 防护空值，避免NullReferenceException
            if (string.IsNullOrEmpty(text))
            {
                return false;
            }

            // LINQ的Any()等价于Python的any()：遍历每个字符，只要有一个满足就返回true
            return text.Any(ch => IsChineseChar(ch));
        }

      

       

        public static void SaveImg(string savePath, Mat img)
        {
            // 空值防护
            if (string.IsNullOrEmpty(savePath))
                throw new ArgumentNullException(nameof(savePath), "The save path cannot be empty");
            if (img == null || img.Empty())
                throw new ArgumentNullException(nameof(img), "The image data cannot be empty");

            // 1. 创建父目录（对应Python：Path(save_path).parent.mkdir(parents=True, exist_ok=True)）
            string parentDir = Path.GetDirectoryName(savePath);
            if (!string.IsNullOrEmpty(parentDir) && !Directory.Exists(parentDir))
            {
                Directory.CreateDirectory(parentDir); // 自动创建所有缺失的父目录，已存在不报错
            }

            // 2. 区分系统保存图片
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var extension = Path.GetExtension(savePath);
                var encoded = img.ImEncode(extension);
                File.WriteAllBytes(savePath, encoded);
            }
            else
            {
                // 非Windows系统：直接用ImWrite（对应Python的cv2.imwrite）
                Cv2.ImWrite(savePath, img);
            }
        }
    }
}
