# 单帧纠正
“Orthophoto Maps”的c++版本。将“Orthophoto Maps”由python转换成c++版本
一种从无人机图像生成个人地图（正射影像）的绘图软件。只有使用图像（和感官数据），您才能生成感兴趣区域的正射影像。
# 安装
项目依赖于proj、opencv、Eigen、gdal
可以通过vcpkg直接安装
`
vcpkg install proj:x64-windows
vcpkg install opencv:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install gdal
`
# 运行
1、输入POS路径
POS格式为照片名称	经度	纬度	绝对高度	横滚角	俯仰角	偏航角
2、参数
主要是焦距、参数、地面高度
