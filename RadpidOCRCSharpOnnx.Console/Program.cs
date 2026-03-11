
using System.Runtime.InteropServices;
namespace RadpidOCRCSharpOnnx.ConsoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var buildNumber = Environment.OSVersion.Version.Build;
            //var dd= NativeMethods.core_getVersionRevision
            Console.WriteLine("Hello, World!");
           
#if DEBUG


            System.Diagnostics.Debug.WriteLine("123123123");
#endif
        }
    }
}
