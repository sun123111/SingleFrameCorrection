#ifndef IMAGE_RECTIFIER_H
#define IMAGE_RECTIFIER_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <tuple>
#include <proj.h>  
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"

namespace fs = std::filesystem;

// 图像纠正器类，封装所有图像纠正相关功能
class ImageRectifier {
private:
    // 常量定义
    const std::string WGS84 = "EPSG:4326";
    const std::string CGCS2000 = "EPSG:4490";

    // 相机参数
    double pixel_size_;    // 像素大小（米）
    double focal_length_;  // 焦距（米）

    // 外方位元素 (x, y, z, omega, phi, kappa)
    Eigen::Matrix<double, 6, 1> eo_;

    // 图像相关
    cv::Mat image_;
    cv::Mat restored_image_;

    // 坐标系统相关
    int epsg_code_;
    std::string target_crs_;

    // 旋转矩阵
    Eigen::Matrix3d rotation_matrix_;

public:
    /**
     * 构造函数
     * @param pixel_size 像素大小（米）
     * @param focal_length 焦距（米）
     */
    ImageRectifier(double pixel_size, double focal_length);

    /**
     * 析构函数
     */
    ~ImageRectifier();

    /**
     * 加载图像
     * @param file_path 图像文件路径
     * @return 是否加载成功
     */
    bool loadImage(const std::string& file_path);

    /**
     * 设置外方位元素
     * @param longitude 经度（度）
     * @param latitude 纬度（度）
     * @param height 高度（米）
     * @param roll 滚转角（度）
     * @param pitch 俯仰角（度）
     * @param yaw 偏航角（度）
     * @param coordinate_system 坐标系（默认WGS84）
     */
    void setExteriorOrientation(double longitude, double latitude, double height,
        double roll, double pitch, double yaw,
        const std::string& coordinate_system = "EPSG:4326");

    /**
     * 恢复图像方向
     * @param orientation 方向参数
     */
    void restoreImageOrientation(int orientation = 1);

    /**
     * 计算图像边界框
     * @param dem 数字高程模型值
     * @return 边界框矩阵
     */
    Eigen::Matrix<double, 4, 1> calculateBoundary(double dem = 0);

    /**
     * 执行平面平行纠正
     * @param boundary 边界框
     * @param gsd 地面采样距离
     * @param ground_height 地面高度
     * @return 纠正后的B、G、R、Alpha通道
     */
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> rectifyPlaneParallel(
        const Eigen::Matrix<double, 4, 1>& boundary, double gsd, double ground_height = 0);

    /**
     * 创建带有地理参考的TIFF文件
     * @param b 蓝色通道
     * @param g 绿色通道
     * @param r 红色通道
     * @param a Alpha通道
     * @param boundary 边界框
     * @param gsd 地面采样距离
     * @param rows 图像高度
     * @param cols 图像宽度
     * @param dst 输出文件路径（不含扩展名）
     */
    void createGeoTiff(const cv::Mat& b, const cv::Mat& g, const cv::Mat& r, const cv::Mat& a,
        const Eigen::Matrix<double, 4, 1>& boundary, double gsd,
        int rows, int cols, const std::string& dst);

    /**
     * 完整的图像纠正流程
     * @param input_path 输入图像路径
     * @param output_dir 输出目录
     * @param dem 数字高程模型值
     * @param orientation 图像方向
     */
    void processImage(const std::string& input_path, const std::string& output_dir,
        double dem = 0, int orientation = 1);

private:
    /**
     * 计算EPSG代码
     * @param longitude 经度
     * @param latitude 纬度
     * @param coordinate_system 坐标系
     * @return EPSG代码
     */
    int epsg_calc(double longitude, double latitude, const std::string& coordinate_system);

    /**
     * 地理坐标转平面坐标
     * @param lon 经度
     * @param lat 纬度
     * @param target_crs 目标坐标系
     * @param source_crs 源坐标系
     * @return 平面坐标
     */
    std::vector<double> geographic2plane(double lon, double lat,
        const std::string& target_crs,
        const std::string& source_crs);

    /**
     * RPY转OPK
     * @param rpy RPY角
     * @param maker 设备制造商
     * @return OPK角
     */
    std::vector<double> rpy_to_opk(const std::vector<double>& rpy, const std::string& maker = "");

    /**
     * 角度转弧度
     * @param degrees 角度值
     * @return 弧度值
     */
    double degrees_to_radians(double degrees);

    /**
     * 2D旋转矩阵
     * @param theta 角度（弧度）
     * @return 2D旋转矩阵
     */
    Eigen::Matrix2d rot2d(double theta);

    /**
     * 3D旋转矩阵
     * @param omega 欧米伽角（弧度）
     * @param phi 菲角（弧度）
     * @param kappa 卡帕角（弧度）
     * @return 3D旋转矩阵
     */
    Eigen::Matrix3d Rot3D(double omega, double phi, double kappa);

    /**
     * 获取图像顶点
     * @return 顶点矩阵
     */
    Eigen::Matrix<double, 3, 4> getVertices();

    /**
     * 投影计算
     * @param vertices 顶点矩阵
     * @param dem 数字高程模型值
     * @return 投影坐标
     */
    Eigen::Matrix<double, 2, 4> projection(const Eigen::Matrix<double, 3, 4>& vertices, double dem);

    /**
     * 旋转图像
     * @param image 输入图像
     * @param angle 旋转角度
     * @return 旋转后的图像
     */
    cv::Mat rotateImage(const cv::Mat& image, double angle);
};

#endif // IMAGE_RECTIFIER_H
