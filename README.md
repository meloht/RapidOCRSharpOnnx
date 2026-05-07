
![RapidOCRSharpOnnx](https://socialify.git.ci/meloht/RapidOCRSharpOnnx/image?description=1&forks=1&issues=1&language=1&name=1&owner=1&pulls=1&stargazers=1&theme=Light)

# RapidOCRSharpOnnx
![PP-OCRv4](https://img.shields.io/badge/PP--OCR-v4-blue)
![PP-OCRv5](https://img.shields.io/badge/PP--OCR-v5-blue)
![C#](https://img.shields.io/badge/language-C%23-blue.svg) 
![.NET Version](https://img.shields.io/badge/dynamic/xml?url=https://raw.githubusercontent.com/meloht/RapidOCRSharpOnnx/refs/heads/master/RapidOCRSharpOnnx/RapidOCRSharpOnnx.csproj&query=//TargetFrameworks&label=.NET)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blue.svg?logo=onnx&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg?logo=opencv&logoColor=white)
![GitHub license](https://img.shields.io/github/license/meloht/RapidOCRSharpOnnx) 
![Release](https://img.shields.io/github/v/release/meloht/RapidOCRSharpOnnx.svg?logo=github&label=Release) 
[![NuGet](https://img.shields.io/nuget/v/RapidOCRSharpOnnx.svg?logo=nuget&logoColor=white)](https://www.nuget.org/packages/RapidOCRSharpOnnx/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/RapidOCRSharpOnnx.svg?logo=nuget)](https://www.nuget.org/packages/RapidOCRSharpOnnx/) 
[![GitHub last commit](https://img.shields.io/github/last-commit/meloht/RapidOCRSharpOnnx?logo=github)](https://github.com/meloht/RapidOCRSharpOnnx)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/meloht/RapidOCRSharpOnnx?logo=github)](https://github.com/meloht/RapidOCRSharpOnnx)

🚀a high performance, cross-platform PaddleOCR C# inference library base on OpenCV and ONNX Runtime.
Referring to the [RapidOCR](https://github.com/RapidAI/RapidOCR) project, it is a python version of the C# implementation with a redesigned and optimized architecture.

# Features
 - **Supported Languages**  PP-OCRv5 provides multilingual text recognition capabilities covering 106 languages, please refer to the documentation: [PP-OCRv5 Multilingual Text Recognition](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html).
 - **Execution Provider** CPU, CUDA / TensorRT, OpenVINO, CoreML, DirectML
 - **Batch processing images** Preprocess and Inference are executed asynchronously  with Producer/Consumer pattern
 - **High Performance Inference** Memory reuse, GPU Inference with I/O Binding
 - **Image Processing** [OpenCvSharp4](https://github.com/shimat/opencvsharp)
 - **Draw Result Image** [SkiaSharp](https://github.com/mono/SkiaSharp)
 - **Inference Engine** [ONNX Runtime](https://github.com/microsoft/onnxruntime) is a cross-platform inference and training machine-learning accelerator.
 - **PP-OCR Versions** Includes support for: [PP-OCRv5](https://www.paddleocr.ai/v3.0.0/version3.x/pipeline_usage/OCR.html), [PP-OCRv4](https://www.paddleocr.ai/v3.0.0/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## Example Images:

<div align="center">
 
| OCR Demo  |
|---------------|
| <img src="https://github.com/meloht/RapidOCRSharpOnnx/blob/master/ExampleImages/res_book.png?raw=true" height="411" > |
| <img src="https://github.com/meloht/RapidOCRSharpOnnx/blob/master/ExampleImages/res_csharp.png?raw=true" height="750"> |
| <img src="https://github.com/meloht/RapidOCRSharpOnnx/blob/master/ExampleImages/res_yongledadian.png?raw=true" height="787"> |

</div>

# Build Package 
Release x64

# Usage
### 1. Export model to ONNX format:
For convert the pre-trained PP-OCR model to ONNX format, please refer to the the documentation: [Obtaining ONNX Models](https://www.paddleocr.ai/v3.0.0/version3.x/deployment/obtaining_onnx_models.html), or download from rapid-ocr [Model List](https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx).

### 2. Load the ONNX model with C#:
Install Nuget packages `RapidOCRSharpOnnx`, `OnnxRuntime`, `OpenCvSharp4.runtime`

#### CPU Inference
```shell
dotnet add package RapidOCRSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Microsoft.ML.OnnxRuntime
```

``` csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCPU(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath)));
```

#### CoreML Inference
```shell
dotnet add package RapidOCRSharpOnnx
dotnet add package OpenCvSharp4.runtime.osx.10.15-x64
dotnet add package Microsoft.ML.OnnxRuntime
```

```csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCoreML(new OcrConfig(detectPath, recPath, LangRec.CH, OCRVersion.PPOCRV5, clsMobilePath)));
```

#### CUDA/TensorRT Inference
```shell
dotnet add package RapidOCRSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Microsoft.ML.OnnxRuntime.Gpu.Windows
```

```csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderCUDA(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), deviceId));
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderTensorRT(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), deviceId));
```

#### DirectML Inference
```shell
dotnet add package RapidOCRSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```

```csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.EN, OCRVersion.PPOCRV5, clsPath), deviceId));
```

#### OpenVINO Inference
```shell
dotnet add package RapidOCRSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Intel.ML.OnnxRuntime.OpenVino
```

```csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderOpenVINO(new OcrConfig(detectPath, recogPath, LangRec.EN, OCRVersion.PPOCRV5, clsPath), IntelDeviceType.NPU));
```

#### Use the following C# code to load the model and run basic prediction

```csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.EN, OCRVersion.PPOCRV5, clsPath), _deviceId));
string savePath = $"res_{Path.GetFileName(imgPath)}";
var result = ocr.RecognizeText(imgPath, savePath);
Console.WriteLine($"result: {result.ToString()}");

```

#### Batch processing images

```csharp

 using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
 var list = Directory.GetFiles(@"C:\code\model\OCRTestImages");
 Stopwatch sw = new Stopwatch();
 sw.Start();
 var resPath = ocr.BatchParallelAsync(list.ToList(), saveDir, receiveAction: ReceiveResult);
 sw.Stop();
 Console.WriteLine($"BatchAsync Time: {sw.ElapsedMilliseconds} ms");

private static void ReceiveResult(OcrBatchResult batchResult)
{
    Console.WriteLine(batchResult.ToString());
    Console.WriteLine("------------------------------------------------------------");
}
```

#### Batch processing images foreach api

```csharp
using RapidOCRSharp ocr = new RapidOCRSharp(new ExecutionProviderDirectML(new OcrConfig(detectPath, recogPath, LangRec.CH, OCRVersion.PPOCRV5, clsPath), _deviceId));
var list = Directory.GetFiles(@"D:\code\model\OCRTestImages");
var resPaths = ocr.BatchForeachAsync(list.ToList(), @"D:\code\model\OCRTestImagesResults");

await foreach (var item in resPaths)
{
    Console.WriteLine(item.TextBlocks);
}

```

# Performance Test

|OCR library|Version|language|Inference Engine|
| ------------- | ------------- | ------------- |------------- |
| [PaddleSharp](https://github.com/sdcb/PaddleSharp)| 3.0.1 |Paddle Inference C API .NET binding|  Sdcb.PaddleInference
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)| 3.5.0 |python| paddlepaddle |
| [RapidOCR](https://github.com/RapidAI/RapidOCR)| 3.8.1 |python| openvino |
| [RapidOCRSharpOnnx](https://github.com/meloht/RapidOCRSharpOnnx)| 1.0.0 |C#|Intel.ML.OnnxRuntime.OpenVino|

## Performance Test PC 

|Hardware|Summary|
| ------------- | ------------- | 
|Windows |Windows 11 Pro OS Version 25H2|
|CPU| Intel Core Ultra 9 285k 3.7GHz|
|RAM| DDR5 128GB speed 4400MT/s|
|Storage| SSD 2TB|

## Performance Test Data

**Images:**  60 images (image size: 1180x92)

**PP-OCR Model:**  ch_PP-OCRv5_det_mobile, ch_PP-OCRv5_rec_mobile, ch_PP-LCNet_x0_25_textline_ori_cls_mobile


## PaddleSharp test result

**CPU inference time:** 48.1769278s 

<img width="1331" height="847" alt="image" src="https://github.com/user-attachments/assets/87515077-e9c2-48b2-9e43-6b145e1c7a7a" />


## RapidOCRSharpOnnx test result

**CPU inference time:** 9.2447871s

<img width="1477" height="874" alt="image" src="https://github.com/user-attachments/assets/eee41ffe-f0a7-48af-b934-74e290ee6196" />


## PaddleOCR test result

**CPU inference time:** 62.668516899924725s 

<img width="1009" height="911" alt="image" src="https://github.com/user-attachments/assets/b367749f-3c37-4326-bb30-ea4fcb52315d" />

## RapidOCR test result

**CPU inference time:** 17.963430000003427 s

<img width="997" height="1093" alt="image" src="https://github.com/user-attachments/assets/79ab7dd5-0311-42ea-aa03-ed5695ff5fae" />


## Performance Test Result

|OCR library|Version|language|Inference Engine|Elapsed Time|
| ------------- | ------------- | ------------- |------------- |------------- |
| [PaddleSharp](https://github.com/sdcb/PaddleSharp)| 3.0.1 |Paddle Inference C API .NET binding|  Sdcb.PaddleInference.runtime.win64.mkl version 3.1.0.54 CPU|48.1769s |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)| 3.5.0 |python| paddlepaddle version 3.2.0 CPU|62.6685s |
| [RapidOCR](https://github.com/RapidAI/RapidOCR)| 3.8.1 |python| openvino version 2026.1.0  21367 CPU|17.9634s|
| [RapidOCRSharpOnnx](https://github.com/meloht/RapidOCRSharpOnnx)| 1.0.0 |C#|Intel.ML.OnnxRuntime.OpenVino CPU  version 1.24.1|9.2447s 
















