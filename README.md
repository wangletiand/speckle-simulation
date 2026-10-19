# 非相干薄相位屏散斑仿真

基于 WaveProp 的空间非相干散斑数据生成。输入目标强度，输出无噪散斑、含噪测量、PSF 和参数。此仓库不包含自相关提取、扩散模型或图像重建。

## 运行

建议 Python 3.10，CPU 即可运行。GPU 版 PyTorch 请按所用 CUDA 环境单独安装。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python read_data.py
python generate.py
```

仓库附带数字 0、圆环和椅子三个样例。generate.py 首次运行下载 MNIST 测试集，生成 16 个目标、3 个散射屏种子、2 个传播距离，共 96 组数据，并重写根目录 manifest.jsonl。运行环境需能访问 MNIST 下载地址。

## 模型

- 波长 532 nm，物体到散射屏 0.3 m，散射屏到相机 0.1/0.2 m。
- 散射屏计算网格 1024×1024，采样间隔 4 μm，圆孔直径 1.2 mm。
- 目标强度 65×65，相机输出 512×512；这是本实现的固定配置，不代表软件通用输入限制。
- WaveProp fresnel_one_step 传播点源经薄相位屏后的复场，取模平方得到强度 PSF。
- 空间非相干物点按强度叠加。在该理想等晕模型下使用线性卷积，再截取传感器区域；负放大率对应目标旋转 180°。
- 每幅图按峰值约 20000 电子调曝光，加入泊松噪声、3 电子读噪声和 100 电子偏置。曝光缩放保存在 NPZ 中；不同图像不是固定曝光条件。

## 数据

| 路径 | 内容 |
| --- | --- |
| targets/ | 目标强度 NPY 和显示 PNG |
| pairs/ | 无噪强度、含噪强度、电子计数和曝光缩放 |
| psfs/ | PSF、相位屏、孔径和参数 JSON |
| manifest.jsonl | 仓库三个样例的对应关系和测量文件 SHA256 |
| docs/reference_manifest.jsonl | 原始 96 组数据记录，完整数据需重新生成 |
| docs/reference_validation.json | 原始运行的数值验证 |
| docs/publication_validation.json | 导出时的独立复跑和逐组核对结果 |

定量处理读取 NPY/NPZ，不用显示图片归一化后的灰度值代替测量。

## 适用范围

空间非相干、单色、标量近轴、零厚度相位屏及理想平移不变模型。未模拟有限厚度导致的角度退相关、实际毛玻璃表面统计、相机像素面积积分或 ADC 饱和量化。仿真不是实际相机和毛玻璃的数字孪生，也不能证明相干激光照明下同样成立。

## 来源

- WaveProp 0.0.12：https://github.com/ebezzam/waveprop （MIT）
- MNIST：https://storage.googleapis.com/cvdf-datasets/mnist/
- 第三方依赖按各自许可证使用；不随仓库打包其安装目录。
