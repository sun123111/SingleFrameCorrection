#include <iostream>
#include "ImageRectifier.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>

// 存储单张图像的参数
struct ImageParams {
    std::string filename;   // 图像文件名
    double longitude;       // 经度
    double latitude;        // 纬度
    double height;          // 高度
    double roll;            // 滚转角
    double pitch;           // 俯仰角
    double yaw;             // 偏航角
};


std::vector<ImageParams> readImageParamsFromTxt(const std::string& file_path) {
    std::vector<ImageParams> params_list;
    std::ifstream file(file_path);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    std::string line;
    int line_number = 0;

    // 逐行读取文件
    while (std::getline(file, line)) {
        line_number++;

        // 跳过空行
        if (line.empty()) {
            continue;
        }

        // 使用字符串流解析行内容
        std::istringstream iss(line);
        ImageParams params;

        // 按照格式提取数据（文件名+7个数值）
        // 注意：分隔符可以是空格或制表符，istringstream会自动处理
        if (!(iss >> params.filename
            >> params.longitude
            >> params.latitude
            >> params.height
            >> params.roll
            >> params.pitch
            >> params.yaw)) {
            // 解析失败时抛出异常，包含行号便于调试
            throw std::runtime_error("文件格式错误，行号: " + std::to_string(line_number)
                + "，内容: " + line);
        }

        params_list.push_back(params);
    }

    // 检查文件是否正常读取结束
    if (file.bad()) {
        throw std::runtime_error("读取文件时发生错误: " + file_path);
    }

    std::cout << "成功读取 " << params_list.size() << " 条图像参数" << std::endl;
    return params_list;
}


bool fileExists(const std::string& file_path) {
    try {
        return fs::exists(file_path) && fs::is_regular_file(file_path);
    }
    catch (const fs::filesystem_error& e) {
        // 处理文件系统错误（如权限问题）
        return false;
    }
}

int main() {
    std::cout << std::setprecision(10) << std::fixed;

    try {
        // 1. 读取TXT文件中的图像参数
        std::string txt_path = "E:\\Slam\\datasets\\08225G\\H20\\Video\\pos.txt"; // 参数文件路径
        std::vector<ImageParams> image_params_list = readImageParamsFromTxt(txt_path);

        // 2. 相机固定参数
        double pixel_size = 0.0196057 / 1000;  // 像素大小（米）
        double focal_length = 27 / 1000.0;     // 焦距（米）
		double ground_height_ = 53;           // 地面高度（米）
        std::string base_image_dir = "E:\\Slam\\datasets\\08225G\\H20\\Video\\out\\"; // 图像所在目录
        std::string output_dir = "E:\\Slam\\datasets\\08225G\\H20\\Video\\out\\"; // 输出目录


        for (auto& params : image_params_list) {
            // 创建图像完整路径
            std::string image_path = base_image_dir + params.filename;
            // 检查图像文件是否存在
            if (!fileExists(image_path)) {
                std::cout << "警告: 图像文件不存在，已跳过 - " << image_path << std::endl;
                continue;
            }
            try {
                // 记录开始时间
                auto start_time = std::chrono::high_resolution_clock::now();
                // 创建纠正器实例
                ImageRectifier rectifier(pixel_size, focal_length, ground_height_);
                // 设置外方位元素
                rectifier.setExteriorOrientation(
                    params.longitude,
                    params.latitude,
                    params.height,
                    params.roll,
                    params.pitch,
                    params.yaw,
                    "EPSG:4326"
                );
                // 执行纠正处理
                rectifier.processImage(image_path, output_dir, 0, 1);
                // 记录结束时间并计算耗时
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_time - start_time;

                // 输出处理结果和耗时，保留两位小数
                std::cout << "处理图像完成: " << params.filename
                    << "，耗时: " << std::fixed << std::setprecision(2)
                    << elapsed.count() << " 秒" << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "处理图像 " << params.filename << " 时出错: " << e.what() << std::endl;
            }
        }
        std::cout << "处理完成！" << std::endl;



    }
    catch (const std::exception& e) {
        std::cerr << "程序错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
