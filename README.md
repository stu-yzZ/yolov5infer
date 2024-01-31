# yolov5infer
#修改demo文件即可
#需要注意得在自己的设备上执行onnx2trt的转换，所以该项目里的trt文件不支持在别的设备上跑通
# windows平台使用 TensorRT 部署 PyTorch 模型

## 前言

自己在网上摸索了很久的基于windows平台的tensorRT部署的问题，终于跑通了，我在找资料的过程中没有找到从思路和细节上都具备的文档，我个人觉得思路最重要，我是很多份拼凑出的思路，顺着思路（有时候会错）一点一点解决问题才得出的结果，所有我打算写这份文档，更加侧重整体逻辑思路，技术细节我理解只要解决思路正确，肯定很多人踩过坑，找一找肯定有的，当然我也会提到。
![image](https://github.com/stu-yzZ/yolov5infer/assets/91549379/f472310a-e10e-4b26-98bd-ea4d15f4ae52)
图片来源：Quick Start Guide :: NVIDIA Deep Learning TensorRT Documentation


##环境：
我是在windows环境下部署的。Win10+3070ti。硬件cuda是11.7。应该和codna环境中的cuda没有关系，但是需要注意一点，有可能硬件设备上没有安装cudnn，虽然codna环境中有，这个细节注意下，我配置vs2019的适合遇到了（网上好多都写了用vs2019配置的，啥思路也不说上来一通秀操作），不过暂时看来vs2019还没用到（我暂时只针对python进行部署）。

## 为什么要部署？

在一些使用场景下，需要将深度学习模型部署到终端设备上，例如智能机器人、自动驾驶等。在这些场景下，需要考虑模型的推理速度和功耗。TensorRT 可以通过量化、融合层次和其他优化技术提高模型的推理速度，从而满足终端设备的性能要求。

##那为什么部署可以解决这个问题？（基于tensorRT）

模型部署到 GPU 上使用的是nVidia专门开发的深度学习推理加速库。这些库专门为 GPU 优化，通过量化、融合层次和其他优化技术提高推理速度。具体有以下：
    	并行性和流水线： GPU 
    	允许并行处理，因此你可以通过同时处理多个输入来提高推理速度。还可以使用流水线技术，确保 GPU 在处理一个输入时已经开始处理下一个输入。
    	硬件特定优化： 针对具体 GPU 架构的优化也是可能的。例如，CUDA 编程可以用于在 NVIDIA GPU 上实现更高级的优化。
而使用模型权重和网络推理使用的框架内置函数，（Pytorch中的forword）。
这两种区别会导致天壤地别的推理速度，我自己看了我设备上跑的模型，有如下的对比。



## 怎么部署？

部署 TensorRT 模型主要分为以下三个步骤：

### 1. 将模型和权重转换为 ONNX 格式

PyTorch 提供了 `torch.onnx.export()` 函数来将模型和权重转换为 ONNX 格式。

```python
import torch

# 导入模型
model = torchvision.models.resnet18()

# 转换模型
torch.onnx.export(
    model,
    torch.randn(1, 3, 224, 224),
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)
