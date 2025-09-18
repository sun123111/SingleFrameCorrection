#include "ImageRectifier.h"

// 构造函数
ImageRectifier::ImageRectifier(double pixel_size, double focal_length, double ground)
    : pixel_size_(pixel_size), focal_length_(focal_length), ground_height_(ground){
    // 初始化外方位元素
    eo_.setZero();
}

// 析构函数
ImageRectifier::~ImageRectifier() {
    // 释放图像资源
    if (!image_.empty()) {
        image_.release();
    }
    if (!restored_image_.empty()) {
        restored_image_.release();
    }
}

// 加载图像
bool ImageRectifier::loadImage(const std::string& file_path) {
    image_ = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    if (image_.empty()) {
        std::cerr << "警告: 无法读取图片 " << file_path << std::endl;
        return false;
    }
    return true;
}

// 设置外方位元素
void ImageRectifier::setExteriorOrientation(double longitude, double latitude, double height,
    double roll, double pitch, double yaw,
    const std::string& coordinate_system) {
    // 存储原始经纬度和高度
    eo_(0) = longitude;
    eo_(1) = latitude;
    eo_(2) = height;

    // 计算EPSG代码
    epsg_code_ = epsg_calc(longitude, latitude, coordinate_system);
    target_crs_ = "EPSG:" + std::to_string(epsg_code_);

    // 将经纬度转换为平面坐标
    std::vector<double> planar_coords = geographic2plane(longitude, latitude, target_crs_, coordinate_system);
    eo_(0) = planar_coords[0];  // X坐标
    eo_(1) = planar_coords[1];  // Y坐标

    // 转换RPY到OPK
    std::vector<double> rpy = { roll, pitch, yaw };
    std::vector<double> opk = rpy_to_opk(rpy);

    // 转换为弧度
    eo_(3) = degrees_to_radians(opk[0]);  // omega
    eo_(4) = degrees_to_radians(opk[1]);  // phi
    eo_(5) = degrees_to_radians(opk[2]);  // kappa

    // 计算旋转矩阵
    rotation_matrix_ = Rot3D(eo_(3), eo_(4), eo_(5));
}

// 恢复图像方向
void ImageRectifier::restoreImageOrientation(int orientation) {
    if (image_.empty()) {
        throw std::runtime_error("未加载图像，请先调用loadImage");
    }
    restored_image_ = rotateImage(image_, 0);  // 初始化为原图

    if (orientation == 8) {
        restored_image_ = rotateImage(image_, -90);
    }
    else if (orientation == 6) {
        restored_image_ = rotateImage(image_, 90);
    }
    else if (orientation == 3) {
        restored_image_ = rotateImage(image_, 180);
    }
}

// 计算图像边界框
Eigen::Matrix<double, 4, 1> ImageRectifier::calculateBoundary(double dem) {
    if (restored_image_.empty()) {
        throw std::runtime_error("未恢复图像方向，请先调用restoreImageOrientation");
    }

    // 旋转矩阵转置
    Eigen::Matrix3d inverse_R = rotation_matrix_.transpose();

    // 获取图像顶点
    Eigen::Matrix<double, 3, 4> image_vertex = getVertices();

    // 投影到地面坐标系
    Eigen::Matrix<double, 2, 4> proj_coordinates = projection(image_vertex, dem);

    // 计算边界框
    Eigen::Matrix<double, 4, 1> bbox;
    bbox(0) = proj_coordinates.row(0).minCoeff();  // X min
    bbox(1) = proj_coordinates.row(0).maxCoeff();  // X max
    bbox(2) = proj_coordinates.row(1).minCoeff();  // Y min
    bbox(3) = proj_coordinates.row(1).maxCoeff();  // Y max

    return bbox;
}

// 执行平面平行纠正
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> ImageRectifier::rectifyPlaneParallel(
    const Eigen::Matrix<double, 4, 1>& boundary, double gsd, double ground_height) {

    if (restored_image_.empty()) {
        throw std::runtime_error("未恢复图像方向，请先调用restoreImageOrientation");
    }

    int boundary_rows = static_cast<int>((boundary(3) - boundary(2)) / gsd);
    int boundary_cols = static_cast<int>((boundary(1) - boundary(0)) / gsd);

    // 初始化输出通道（B, G, R, Alpha）
    cv::Mat b(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat g(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat r(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat a(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));

    // 获取输入图像尺寸
    int image_rows = restored_image_.rows;
    int image_cols = restored_image_.cols;
    int image_half_rows = image_rows / 2;
    int image_half_cols = image_cols / 2;

    // 并行处理行
#pragma omp parallel for collapse(2)
    for (int row = 0; row < boundary_rows; ++row) {
        for (int col = 0; col < boundary_cols; ++col) {
            // 投影计算
            double proj_coords_x = boundary(0) + col * gsd - eo_(0);
            double proj_coords_y = boundary(3) - row * gsd - eo_(1);
            double proj_coords_z = ground_height - eo_(2);

            // 反投影计算（单位：米）
            double coord_CCS_m_x = rotation_matrix_(0, 0) * proj_coords_x +
                rotation_matrix_(0, 1) * proj_coords_y +
                rotation_matrix_(0, 2) * proj_coords_z;
            double coord_CCS_m_y = rotation_matrix_(1, 0) * proj_coords_x +
                rotation_matrix_(1, 1) * proj_coords_y +
                rotation_matrix_(1, 2) * proj_coords_z;
            double coord_CCS_m_z = rotation_matrix_(2, 0) * proj_coords_x +
                rotation_matrix_(2, 1) * proj_coords_y +
                rotation_matrix_(2, 2) * proj_coords_z;

            // 计算缩放因子
            double scale = coord_CCS_m_z / (-focal_length_);
            double plane_coord_CCS_x = coord_CCS_m_x / scale;
            double plane_coord_CCS_y = coord_CCS_m_y / scale;

            // 转换到像素坐标系（单位：像素）
            double coord_CCS_px_x = plane_coord_CCS_x / pixel_size_;
            double coord_CCS_px_y = -plane_coord_CCS_y / pixel_size_;

            // 重采样（最近邻插值）
            int coord_ICS_col = static_cast<int>(image_half_cols + coord_CCS_px_x);
            int coord_ICS_row = static_cast<int>(image_half_rows + coord_CCS_px_y);

            // 检查是否在图像范围内
            if (coord_ICS_col >= 0 && coord_ICS_col < image_cols &&
                coord_ICS_row >= 0 && coord_ICS_row < image_rows) {

                // 读取BGR通道
                cv::Vec3b pixel = restored_image_.at<cv::Vec3b>(coord_ICS_row, coord_ICS_col);
                b.at<uchar>(row, col) = pixel[0];  // B通道
                g.at<uchar>(row, col) = pixel[1];  // G通道
                r.at<uchar>(row, col) = pixel[2];  // R通道
                a.at<uchar>(row, col) = 255;       // Alpha通道
            }
        }
    }

    return std::make_tuple(b, g, r, a);
}

// 创建带有地理参考的TIFF文件
void ImageRectifier::createGeoTiff(const cv::Mat& b, const cv::Mat& g, const cv::Mat& r, const cv::Mat& a,
    const Eigen::Matrix<double, 4, 1>& boundary, double gsd,
    int rows, int cols, const std::string& dst) {
    // 注册GDAL驱动
    GDALAllRegister();

    // 输出文件路径
    std::string output_path = dst + ".tif";

    // 获取TIFF驱动
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == nullptr) {
        throw std::runtime_error("无法获取GTiff驱动");
    }

    // 创建输出数据集
    GDALDataset* poDstDS = poDriver->Create(
        output_path.c_str(),
        cols,
        rows,
        4,
        GDT_Byte,
        nullptr
    );
    if (poDstDS == nullptr) {
        throw std::runtime_error("无法创建输出TIFF文件: " + output_path);
    }

    // 设置地理变换参数
    double adfGeoTransform[6] = {
        boundary(0),  // 左上角X坐标
        gsd,          // X方向分辨率
        0,            // X方向旋转
        boundary(3),  // 左上角Y坐标
        0,            // Y方向旋转
        -gsd          // Y方向分辨率
    };
    poDstDS->SetGeoTransform(adfGeoTransform);

    // 设置投影信息
    OGRSpatialReference oSRS;
    if (oSRS.importFromEPSG(epsg_code_) != OGRERR_NONE) {
        throw std::runtime_error("无法导入EPSG编码: " + std::to_string(epsg_code_));
    }
    char* pszWKT = nullptr;
    oSRS.exportToWkt(&pszWKT);
    poDstDS->SetProjection(pszWKT);
    CPLFree(pszWKT);

    // 写入各波段数据
    GDALRasterBand* poBand1 = poDstDS->GetRasterBand(1);
    poBand1->RasterIO(GF_Write, 0, 0, cols, rows, r.data, cols, rows, GDT_Byte, 0, 0);

    GDALRasterBand* poBand2 = poDstDS->GetRasterBand(2);
    poBand2->RasterIO(GF_Write, 0, 0, cols, rows, g.data, cols, rows, GDT_Byte, 0, 0);

    GDALRasterBand* poBand3 = poDstDS->GetRasterBand(3);
    poBand3->RasterIO(GF_Write, 0, 0, cols, rows, b.data, cols, rows, GDT_Byte, 0, 0);

    GDALRasterBand* poBand4 = poDstDS->GetRasterBand(4);
    poBand4->RasterIO(GF_Write, 0, 0, cols, rows, a.data, cols, rows, GDT_Byte, 0, 0);

    // 释放资源
    GDALClose((GDALDatasetH)poDstDS);
}

// 完整的图像纠正流程
void ImageRectifier::processImage(const std::string& input_path, const std::string& output_dir,
    double dem, int orientation) {
    try {
        // 1. 加载图像
        if (!loadImage(input_path)) {
            return;
        }

        // 2. 恢复图像方向
        restoreImageOrientation(orientation);

        // 3. 计算相对高度
        eo_(2) -= ground_height_;  // 假设地面高度为53米

        // 4. 计算边界框
        Eigen::Matrix<double, 4, 1> bbox = calculateBoundary(dem);

        // 5. 计算GSD
        double gsd = (pixel_size_ * eo_(2)) / focal_length_;

        // 6. 计算输出图像尺寸
        int boundary_cols = static_cast<int>((bbox(1) - bbox(0)) / gsd);
        int boundary_rows = static_cast<int>((bbox(3) - bbox(2)) / gsd);

        // 7. 执行平面平行纠正
        auto [b, g, r, a] = rectifyPlaneParallel(bbox, gsd, dem);

        // 8. 准备输出路径
        fs::path output_path = fs::path(output_dir) / "rectified";
        fs::create_directories(output_path);

        std::string filename = fs::path(input_path).stem().string();
        fs::path dst = output_path / filename;

        // 9. 创建GeoTIFF
        createGeoTiff(b, g, r, a, bbox, gsd, boundary_rows, boundary_cols, dst.string());

        // 10. 释放资源
        image_.release();
        restored_image_.release();

    }
    catch (const std::exception& e) {
        std::cerr << "处理图像时出错: " << e.what() << std::endl;
        throw;  // 重新抛出异常，让调用者也能捕获
    }
}

// 计算EPSG代码
int ImageRectifier::epsg_calc(double longitude, double latitude, const std::string& coordinate_system) {
    // 纬度范围检查
    if (latitude < -80.0 || latitude > 84.0) {
        throw std::invalid_argument("UTM坐标系只在纬度 -80 到 84 度之间有效");
    }

    // 经度范围检查
    if (longitude < -180.0 || longitude > 180.0) {
        throw std::invalid_argument("经度应该在 -180 到 180 度之间");
    }

    if (coordinate_system == WGS84) {
        // 计算UTM带号
        int zone_number = static_cast<int>((longitude + 180) / 6) + 1;

        // 判断南北半球
        if (latitude >= 0) {
            return 32600 + zone_number;  // 北半球
        }
        else {
            return 32700 + zone_number;  // 南半球
        }
    }
    else if (coordinate_system == CGCS2000) {
        // 计算CGCS2000带号
        int band_number = static_cast<int>((longitude - 1.5) / 3) + 1;
        return 4534 + (band_number - 25);
    }
    else {
        throw std::invalid_argument("不支持的坐标系类型。支持的类型有 'WGS84' 和 'CGCS2000'");
    }
}

// 地理坐标转平面坐标
std::vector<double> ImageRectifier::geographic2plane(double lon, double lat,
    const std::string& target_crs,
    const std::string& source_crs) {
    // 输入参数校验
    if (lat < -90.0 || lat > 90.0) {
        throw std::invalid_argument("纬度必须在 -90 到 90 度之间");
    }
    if (lon < -180.0 || lon > 180.0) {
        throw std::invalid_argument("经度必须在 -180 到 180 度之间");
    }

    // 初始化PROJ转换上下文
    PJ_CONTEXT* ctx = proj_context_create();
    if (!ctx) {
        throw std::runtime_error("无法创建PROJ上下文");
    }

    // 创建转换对象
    PJ* transformer = proj_create_crs_to_crs(ctx, source_crs.c_str(), target_crs.c_str(), nullptr);
    if (!transformer) {
        const char* err_msg = proj_errno_string(proj_context_errno(ctx));
        proj_context_destroy(ctx);
        throw std::runtime_error("转换对象创建失败: " + std::string(err_msg ? err_msg : "未知错误"));
    }

    // 执行坐标转换
    PJ_COORD input = proj_coord(lat, lon, 0.0, 0.0);
    PJ_COORD output = proj_trans(transformer, PJ_FWD, input);

    // 检查转换结果
    if (std::isnan(output.xy.x) || std::isnan(output.xy.y) ||
        std::isinf(output.xy.x) || std::isinf(output.xy.y)) {
        const char* err_msg = proj_errno_string(proj_errno(transformer));
        proj_destroy(transformer);
        proj_context_destroy(ctx);
        throw std::runtime_error("转换失败: " + std::string(err_msg ? err_msg : "结果为无穷大或NaN"));
    }

    // 清理资源
    proj_destroy(transformer);
    proj_context_destroy(ctx);

    return { output.xy.x, output.xy.y };
}

// RPY转OPK
std::vector<double> ImageRectifier::rpy_to_opk(const std::vector<double>& rpy, const std::string& maker) {
    if (rpy.size() < 3) {
        throw std::invalid_argument("rpy向量必须至少包含3个元素");
    }

    Eigen::Vector2d roll_pitch;

    if (maker == "samsung") {
        roll_pitch(0) = -rpy[1];
        roll_pitch(1) = -rpy[0];
    }
    else {
        roll_pitch(0) = 90.0 + rpy[1];

        if (180.0 - std::abs(rpy[0]) <= 0.1) {
            roll_pitch(1) = 0.0;
        }
        else {
            roll_pitch(1) = rpy[0];
        }
    }

    // 计算旋转矩阵与向量的乘积
    double theta = degrees_to_radians(rpy[2]);
    Eigen::Matrix2d rotation = rot2d(theta);
    Eigen::Vector2d omega_phi = rotation * roll_pitch;

    // 计算kappa值
    double kappa;
    if (maker == "samsung") {
        kappa = -rpy[2] - 90.0;
    }
    else {
        kappa = -rpy[2];
    }

    return { omega_phi(0), omega_phi(1), kappa };
}

// 角度转弧度
double ImageRectifier::degrees_to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}

// 2D旋转矩阵
Eigen::Matrix2d ImageRectifier::rot2d(double theta) {
    Eigen::Matrix2d rot;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    rot << cos_theta, sin_theta,
        -sin_theta, cos_theta;
    return rot;
}

// 3D旋转矩阵
Eigen::Matrix3d ImageRectifier::Rot3D(double omega, double phi, double kappa) {
    double om = omega;
    double ph = phi;
    double kp = kappa;

    // 计算Rx矩阵
    Eigen::Matrix3d Rx = Eigen::Matrix3d::Zero();
    double cos_om = std::cos(om);
    double sin_om = std::sin(om);
    Rx(0, 0) = 1.0;
    Rx(1, 1) = cos_om;
    Rx(1, 2) = sin_om;
    Rx(2, 1) = -sin_om;
    Rx(2, 2) = cos_om;

    // 计算Ry矩阵
    Eigen::Matrix3d Ry = Eigen::Matrix3d::Zero();
    double cos_ph = std::cos(ph);
    double sin_ph = std::sin(ph);
    Ry(0, 0) = cos_ph;
    Ry(0, 2) = -sin_ph;
    Ry(1, 1) = 1.0;
    Ry(2, 0) = sin_ph;
    Ry(2, 2) = cos_ph;

    // 计算Rz矩阵
    Eigen::Matrix3d Rz = Eigen::Matrix3d::Zero();
    double cos_kp = std::cos(kp);
    double sin_kp = std::sin(kp);
    Rz(0, 0) = cos_kp;
    Rz(0, 1) = sin_kp;
    Rz(1, 0) = -sin_kp;
    Rz(1, 1) = cos_kp;
    Rz(2, 2) = 1.0;

    // 计算最终旋转矩阵
    return Rz * Ry * Rx;
}

// 获取图像顶点
Eigen::Matrix<double, 3, 4> ImageRectifier::getVertices() {
    if (restored_image_.empty()) {
        throw std::invalid_argument("输入图像为空！");
    }

    int rows = restored_image_.rows;
    int cols = restored_image_.cols;

    Eigen::Matrix<double, 3, 4> vertices;

    // 四个顶点坐标
    vertices(0, 0) = -cols * pixel_size_ / 2.0;
    vertices(1, 0) = rows * pixel_size_ / 2.0;
    vertices(2, 0) = -focal_length_;

    vertices(0, 1) = cols * pixel_size_ / 2.0;
    vertices(1, 1) = rows * pixel_size_ / 2.0;
    vertices(2, 1) = -focal_length_;

    vertices(0, 2) = cols * pixel_size_ / 2.0;
    vertices(1, 2) = -rows * pixel_size_ / 2.0;
    vertices(2, 2) = -focal_length_;

    vertices(0, 3) = -cols * pixel_size_ / 2.0;
    vertices(1, 3) = -rows * pixel_size_ / 2.0;
    vertices(2, 3) = -focal_length_;

    return vertices;
}

// 投影计算
Eigen::Matrix<double, 2, 4> ImageRectifier::projection(const Eigen::Matrix<double, 3, 4>& vertices, double dem) {
    // 旋转矩阵转置乘以顶点矩阵
    Eigen::Matrix<double, 3, 4> coord_GCS = rotation_matrix_.transpose() * vertices;

    // 计算比例因子
    Eigen::Matrix<double, 1, 4> scale = (dem - eo_(2)) * coord_GCS.row(2).cwiseInverse();

    // 计算平面坐标
    Eigen::Matrix<double, 2, 4> plane_coord_GCS;
    plane_coord_GCS.row(0) = scale.array() * coord_GCS.row(0).array() + eo_(0);
    plane_coord_GCS.row(1) = scale.array() * coord_GCS.row(1).array() + eo_(1);

    return plane_coord_GCS;
}

// 旋转图像
cv::Mat ImageRectifier::rotateImage(const cv::Mat& image, double angle) {
    int height = image.rows;
    int width = image.cols;
    cv::Point2f center(width / 2.0, height / 2.0);

    // 获取旋转矩阵
    cv::Mat rotation_mat = cv::getRotationMatrix2D(center, angle, 1.0);

    // 计算旋转后的图像尺寸
    double abs_cos = std::abs(rotation_mat.at<double>(0, 0));
    double abs_sin = std::abs(rotation_mat.at<double>(0, 1));

    int bound_w = static_cast<int>(height * abs_sin + width * abs_cos);
    int bound_h = static_cast<int>(height * abs_cos + width * abs_sin);

    // 调整旋转矩阵
    rotation_mat.at<double>(0, 2) += bound_w / 2.0 - center.x;
    rotation_mat.at<double>(1, 2) += bound_h / 2.0 - center.y;

    // 执行旋转
    cv::Mat rotated_image;
    cv::warpAffine(image, rotated_image, rotation_mat, cv::Size(bound_w, bound_h));
    return rotated_image;
}
