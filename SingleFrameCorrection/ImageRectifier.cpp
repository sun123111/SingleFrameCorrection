#include "ImageRectifier.h"

// ���캯��
ImageRectifier::ImageRectifier(double pixel_size, double focal_length, double ground)
    : pixel_size_(pixel_size), focal_length_(focal_length), ground_height_(ground){
    // ��ʼ���ⷽλԪ��
    eo_.setZero();
}

// ��������
ImageRectifier::~ImageRectifier() {
    // �ͷ�ͼ����Դ
    if (!image_.empty()) {
        image_.release();
    }
    if (!restored_image_.empty()) {
        restored_image_.release();
    }
}

// ����ͼ��
bool ImageRectifier::loadImage(const std::string& file_path) {
    image_ = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    if (image_.empty()) {
        std::cerr << "����: �޷���ȡͼƬ " << file_path << std::endl;
        return false;
    }
    return true;
}

// �����ⷽλԪ��
void ImageRectifier::setExteriorOrientation(double longitude, double latitude, double height,
    double roll, double pitch, double yaw,
    const std::string& coordinate_system) {
    // �洢ԭʼ��γ�Ⱥ͸߶�
    eo_(0) = longitude;
    eo_(1) = latitude;
    eo_(2) = height;

    // ����EPSG����
    epsg_code_ = epsg_calc(longitude, latitude, coordinate_system);
    target_crs_ = "EPSG:" + std::to_string(epsg_code_);

    // ����γ��ת��Ϊƽ������
    std::vector<double> planar_coords = geographic2plane(longitude, latitude, target_crs_, coordinate_system);
    eo_(0) = planar_coords[0];  // X����
    eo_(1) = planar_coords[1];  // Y����

    // ת��RPY��OPK
    std::vector<double> rpy = { roll, pitch, yaw };
    std::vector<double> opk = rpy_to_opk(rpy);

    // ת��Ϊ����
    eo_(3) = degrees_to_radians(opk[0]);  // omega
    eo_(4) = degrees_to_radians(opk[1]);  // phi
    eo_(5) = degrees_to_radians(opk[2]);  // kappa

    // ������ת����
    rotation_matrix_ = Rot3D(eo_(3), eo_(4), eo_(5));
}

// �ָ�ͼ����
void ImageRectifier::restoreImageOrientation(int orientation) {
    if (image_.empty()) {
        throw std::runtime_error("δ����ͼ�����ȵ���loadImage");
    }
    restored_image_ = rotateImage(image_, 0);  // ��ʼ��Ϊԭͼ

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

// ����ͼ��߽��
Eigen::Matrix<double, 4, 1> ImageRectifier::calculateBoundary(double dem) {
    if (restored_image_.empty()) {
        throw std::runtime_error("δ�ָ�ͼ�������ȵ���restoreImageOrientation");
    }

    // ��ת����ת��
    Eigen::Matrix3d inverse_R = rotation_matrix_.transpose();

    // ��ȡͼ�񶥵�
    Eigen::Matrix<double, 3, 4> image_vertex = getVertices();

    // ͶӰ����������ϵ
    Eigen::Matrix<double, 2, 4> proj_coordinates = projection(image_vertex, dem);

    // ����߽��
    Eigen::Matrix<double, 4, 1> bbox;
    bbox(0) = proj_coordinates.row(0).minCoeff();  // X min
    bbox(1) = proj_coordinates.row(0).maxCoeff();  // X max
    bbox(2) = proj_coordinates.row(1).minCoeff();  // Y min
    bbox(3) = proj_coordinates.row(1).maxCoeff();  // Y max

    return bbox;
}

// ִ��ƽ��ƽ�о���
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> ImageRectifier::rectifyPlaneParallel(
    const Eigen::Matrix<double, 4, 1>& boundary, double gsd, double ground_height) {

    if (restored_image_.empty()) {
        throw std::runtime_error("δ�ָ�ͼ�������ȵ���restoreImageOrientation");
    }

    int boundary_rows = static_cast<int>((boundary(3) - boundary(2)) / gsd);
    int boundary_cols = static_cast<int>((boundary(1) - boundary(0)) / gsd);

    // ��ʼ�����ͨ����B, G, R, Alpha��
    cv::Mat b(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat g(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat r(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat a(boundary_rows, boundary_cols, CV_8UC1, cv::Scalar(0));

    // ��ȡ����ͼ��ߴ�
    int image_rows = restored_image_.rows;
    int image_cols = restored_image_.cols;
    int image_half_rows = image_rows / 2;
    int image_half_cols = image_cols / 2;

    // ���д�����
#pragma omp parallel for collapse(2)
    for (int row = 0; row < boundary_rows; ++row) {
        for (int col = 0; col < boundary_cols; ++col) {
            // ͶӰ����
            double proj_coords_x = boundary(0) + col * gsd - eo_(0);
            double proj_coords_y = boundary(3) - row * gsd - eo_(1);
            double proj_coords_z = ground_height - eo_(2);

            // ��ͶӰ���㣨��λ���ף�
            double coord_CCS_m_x = rotation_matrix_(0, 0) * proj_coords_x +
                rotation_matrix_(0, 1) * proj_coords_y +
                rotation_matrix_(0, 2) * proj_coords_z;
            double coord_CCS_m_y = rotation_matrix_(1, 0) * proj_coords_x +
                rotation_matrix_(1, 1) * proj_coords_y +
                rotation_matrix_(1, 2) * proj_coords_z;
            double coord_CCS_m_z = rotation_matrix_(2, 0) * proj_coords_x +
                rotation_matrix_(2, 1) * proj_coords_y +
                rotation_matrix_(2, 2) * proj_coords_z;

            // ������������
            double scale = coord_CCS_m_z / (-focal_length_);
            double plane_coord_CCS_x = coord_CCS_m_x / scale;
            double plane_coord_CCS_y = coord_CCS_m_y / scale;

            // ת������������ϵ����λ�����أ�
            double coord_CCS_px_x = plane_coord_CCS_x / pixel_size_;
            double coord_CCS_px_y = -plane_coord_CCS_y / pixel_size_;

            // �ز���������ڲ�ֵ��
            int coord_ICS_col = static_cast<int>(image_half_cols + coord_CCS_px_x);
            int coord_ICS_row = static_cast<int>(image_half_rows + coord_CCS_px_y);

            // ����Ƿ���ͼ��Χ��
            if (coord_ICS_col >= 0 && coord_ICS_col < image_cols &&
                coord_ICS_row >= 0 && coord_ICS_row < image_rows) {

                // ��ȡBGRͨ��
                cv::Vec3b pixel = restored_image_.at<cv::Vec3b>(coord_ICS_row, coord_ICS_col);
                b.at<uchar>(row, col) = pixel[0];  // Bͨ��
                g.at<uchar>(row, col) = pixel[1];  // Gͨ��
                r.at<uchar>(row, col) = pixel[2];  // Rͨ��
                a.at<uchar>(row, col) = 255;       // Alphaͨ��
            }
        }
    }

    return std::make_tuple(b, g, r, a);
}

// �������е���ο���TIFF�ļ�
void ImageRectifier::createGeoTiff(const cv::Mat& b, const cv::Mat& g, const cv::Mat& r, const cv::Mat& a,
    const Eigen::Matrix<double, 4, 1>& boundary, double gsd,
    int rows, int cols, const std::string& dst) {
    // ע��GDAL����
    GDALAllRegister();

    // ����ļ�·��
    std::string output_path = dst + ".tif";

    // ��ȡTIFF����
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == nullptr) {
        throw std::runtime_error("�޷���ȡGTiff����");
    }

    // ����������ݼ�
    GDALDataset* poDstDS = poDriver->Create(
        output_path.c_str(),
        cols,
        rows,
        4,
        GDT_Byte,
        nullptr
    );
    if (poDstDS == nullptr) {
        throw std::runtime_error("�޷��������TIFF�ļ�: " + output_path);
    }

    // ���õ���任����
    double adfGeoTransform[6] = {
        boundary(0),  // ���Ͻ�X����
        gsd,          // X����ֱ���
        0,            // X������ת
        boundary(3),  // ���Ͻ�Y����
        0,            // Y������ת
        -gsd          // Y����ֱ���
    };
    poDstDS->SetGeoTransform(adfGeoTransform);

    // ����ͶӰ��Ϣ
    OGRSpatialReference oSRS;
    if (oSRS.importFromEPSG(epsg_code_) != OGRERR_NONE) {
        throw std::runtime_error("�޷�����EPSG����: " + std::to_string(epsg_code_));
    }
    char* pszWKT = nullptr;
    oSRS.exportToWkt(&pszWKT);
    poDstDS->SetProjection(pszWKT);
    CPLFree(pszWKT);

    // д�����������
    GDALRasterBand* poBand1 = poDstDS->GetRasterBand(1);
    poBand1->RasterIO(GF_Write, 0, 0, cols, rows, r.data, cols, rows, GDT_Byte, 0, 0);

    GDALRasterBand* poBand2 = poDstDS->GetRasterBand(2);
    poBand2->RasterIO(GF_Write, 0, 0, cols, rows, g.data, cols, rows, GDT_Byte, 0, 0);

    GDALRasterBand* poBand3 = poDstDS->GetRasterBand(3);
    poBand3->RasterIO(GF_Write, 0, 0, cols, rows, b.data, cols, rows, GDT_Byte, 0, 0);

    GDALRasterBand* poBand4 = poDstDS->GetRasterBand(4);
    poBand4->RasterIO(GF_Write, 0, 0, cols, rows, a.data, cols, rows, GDT_Byte, 0, 0);

    // �ͷ���Դ
    GDALClose((GDALDatasetH)poDstDS);
}

// ������ͼ���������
void ImageRectifier::processImage(const std::string& input_path, const std::string& output_dir,
    double dem, int orientation) {
    try {
        // 1. ����ͼ��
        if (!loadImage(input_path)) {
            return;
        }

        // 2. �ָ�ͼ����
        restoreImageOrientation(orientation);

        // 3. ������Ը߶�
        eo_(2) -= ground_height_;  // �������߶�Ϊ53��

        // 4. ����߽��
        Eigen::Matrix<double, 4, 1> bbox = calculateBoundary(dem);

        // 5. ����GSD
        double gsd = (pixel_size_ * eo_(2)) / focal_length_;

        // 6. �������ͼ��ߴ�
        int boundary_cols = static_cast<int>((bbox(1) - bbox(0)) / gsd);
        int boundary_rows = static_cast<int>((bbox(3) - bbox(2)) / gsd);

        // 7. ִ��ƽ��ƽ�о���
        auto [b, g, r, a] = rectifyPlaneParallel(bbox, gsd, dem);

        // 8. ׼�����·��
        fs::path output_path = fs::path(output_dir) / "rectified";
        fs::create_directories(output_path);

        std::string filename = fs::path(input_path).stem().string();
        fs::path dst = output_path / filename;

        // 9. ����GeoTIFF
        createGeoTiff(b, g, r, a, bbox, gsd, boundary_rows, boundary_cols, dst.string());

        // 10. �ͷ���Դ
        image_.release();
        restored_image_.release();

    }
    catch (const std::exception& e) {
        std::cerr << "����ͼ��ʱ����: " << e.what() << std::endl;
        throw;  // �����׳��쳣���õ�����Ҳ�ܲ���
    }
}

// ����EPSG����
int ImageRectifier::epsg_calc(double longitude, double latitude, const std::string& coordinate_system) {
    // γ�ȷ�Χ���
    if (latitude < -80.0 || latitude > 84.0) {
        throw std::invalid_argument("UTM����ϵֻ��γ�� -80 �� 84 ��֮����Ч");
    }

    // ���ȷ�Χ���
    if (longitude < -180.0 || longitude > 180.0) {
        throw std::invalid_argument("����Ӧ���� -180 �� 180 ��֮��");
    }

    if (coordinate_system == WGS84) {
        // ����UTM����
        int zone_number = static_cast<int>((longitude + 180) / 6) + 1;

        // �ж��ϱ�����
        if (latitude >= 0) {
            return 32600 + zone_number;  // ������
        }
        else {
            return 32700 + zone_number;  // �ϰ���
        }
    }
    else if (coordinate_system == CGCS2000) {
        // ����CGCS2000����
        int band_number = static_cast<int>((longitude - 1.5) / 3) + 1;
        return 4534 + (band_number - 25);
    }
    else {
        throw std::invalid_argument("��֧�ֵ�����ϵ���͡�֧�ֵ������� 'WGS84' �� 'CGCS2000'");
    }
}

// ��������תƽ������
std::vector<double> ImageRectifier::geographic2plane(double lon, double lat,
    const std::string& target_crs,
    const std::string& source_crs) {
    // �������У��
    if (lat < -90.0 || lat > 90.0) {
        throw std::invalid_argument("γ�ȱ����� -90 �� 90 ��֮��");
    }
    if (lon < -180.0 || lon > 180.0) {
        throw std::invalid_argument("���ȱ����� -180 �� 180 ��֮��");
    }

    // ��ʼ��PROJת��������
    PJ_CONTEXT* ctx = proj_context_create();
    if (!ctx) {
        throw std::runtime_error("�޷�����PROJ������");
    }

    // ����ת������
    PJ* transformer = proj_create_crs_to_crs(ctx, source_crs.c_str(), target_crs.c_str(), nullptr);
    if (!transformer) {
        const char* err_msg = proj_errno_string(proj_context_errno(ctx));
        proj_context_destroy(ctx);
        throw std::runtime_error("ת�����󴴽�ʧ��: " + std::string(err_msg ? err_msg : "δ֪����"));
    }

    // ִ������ת��
    PJ_COORD input = proj_coord(lat, lon, 0.0, 0.0);
    PJ_COORD output = proj_trans(transformer, PJ_FWD, input);

    // ���ת�����
    if (std::isnan(output.xy.x) || std::isnan(output.xy.y) ||
        std::isinf(output.xy.x) || std::isinf(output.xy.y)) {
        const char* err_msg = proj_errno_string(proj_errno(transformer));
        proj_destroy(transformer);
        proj_context_destroy(ctx);
        throw std::runtime_error("ת��ʧ��: " + std::string(err_msg ? err_msg : "���Ϊ������NaN"));
    }

    // ������Դ
    proj_destroy(transformer);
    proj_context_destroy(ctx);

    return { output.xy.x, output.xy.y };
}

// RPYתOPK
std::vector<double> ImageRectifier::rpy_to_opk(const std::vector<double>& rpy, const std::string& maker) {
    if (rpy.size() < 3) {
        throw std::invalid_argument("rpy�����������ٰ���3��Ԫ��");
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

    // ������ת�����������ĳ˻�
    double theta = degrees_to_radians(rpy[2]);
    Eigen::Matrix2d rotation = rot2d(theta);
    Eigen::Vector2d omega_phi = rotation * roll_pitch;

    // ����kappaֵ
    double kappa;
    if (maker == "samsung") {
        kappa = -rpy[2] - 90.0;
    }
    else {
        kappa = -rpy[2];
    }

    return { omega_phi(0), omega_phi(1), kappa };
}

// �Ƕ�ת����
double ImageRectifier::degrees_to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}

// 2D��ת����
Eigen::Matrix2d ImageRectifier::rot2d(double theta) {
    Eigen::Matrix2d rot;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    rot << cos_theta, sin_theta,
        -sin_theta, cos_theta;
    return rot;
}

// 3D��ת����
Eigen::Matrix3d ImageRectifier::Rot3D(double omega, double phi, double kappa) {
    double om = omega;
    double ph = phi;
    double kp = kappa;

    // ����Rx����
    Eigen::Matrix3d Rx = Eigen::Matrix3d::Zero();
    double cos_om = std::cos(om);
    double sin_om = std::sin(om);
    Rx(0, 0) = 1.0;
    Rx(1, 1) = cos_om;
    Rx(1, 2) = sin_om;
    Rx(2, 1) = -sin_om;
    Rx(2, 2) = cos_om;

    // ����Ry����
    Eigen::Matrix3d Ry = Eigen::Matrix3d::Zero();
    double cos_ph = std::cos(ph);
    double sin_ph = std::sin(ph);
    Ry(0, 0) = cos_ph;
    Ry(0, 2) = -sin_ph;
    Ry(1, 1) = 1.0;
    Ry(2, 0) = sin_ph;
    Ry(2, 2) = cos_ph;

    // ����Rz����
    Eigen::Matrix3d Rz = Eigen::Matrix3d::Zero();
    double cos_kp = std::cos(kp);
    double sin_kp = std::sin(kp);
    Rz(0, 0) = cos_kp;
    Rz(0, 1) = sin_kp;
    Rz(1, 0) = -sin_kp;
    Rz(1, 1) = cos_kp;
    Rz(2, 2) = 1.0;

    // ����������ת����
    return Rz * Ry * Rx;
}

// ��ȡͼ�񶥵�
Eigen::Matrix<double, 3, 4> ImageRectifier::getVertices() {
    if (restored_image_.empty()) {
        throw std::invalid_argument("����ͼ��Ϊ�գ�");
    }

    int rows = restored_image_.rows;
    int cols = restored_image_.cols;

    Eigen::Matrix<double, 3, 4> vertices;

    // �ĸ���������
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

// ͶӰ����
Eigen::Matrix<double, 2, 4> ImageRectifier::projection(const Eigen::Matrix<double, 3, 4>& vertices, double dem) {
    // ��ת����ת�ó��Զ������
    Eigen::Matrix<double, 3, 4> coord_GCS = rotation_matrix_.transpose() * vertices;

    // �����������
    Eigen::Matrix<double, 1, 4> scale = (dem - eo_(2)) * coord_GCS.row(2).cwiseInverse();

    // ����ƽ������
    Eigen::Matrix<double, 2, 4> plane_coord_GCS;
    plane_coord_GCS.row(0) = scale.array() * coord_GCS.row(0).array() + eo_(0);
    plane_coord_GCS.row(1) = scale.array() * coord_GCS.row(1).array() + eo_(1);

    return plane_coord_GCS;
}

// ��תͼ��
cv::Mat ImageRectifier::rotateImage(const cv::Mat& image, double angle) {
    int height = image.rows;
    int width = image.cols;
    cv::Point2f center(width / 2.0, height / 2.0);

    // ��ȡ��ת����
    cv::Mat rotation_mat = cv::getRotationMatrix2D(center, angle, 1.0);

    // ������ת���ͼ��ߴ�
    double abs_cos = std::abs(rotation_mat.at<double>(0, 0));
    double abs_sin = std::abs(rotation_mat.at<double>(0, 1));

    int bound_w = static_cast<int>(height * abs_sin + width * abs_cos);
    int bound_h = static_cast<int>(height * abs_cos + width * abs_sin);

    // ������ת����
    rotation_mat.at<double>(0, 2) += bound_w / 2.0 - center.x;
    rotation_mat.at<double>(1, 2) += bound_h / 2.0 - center.y;

    // ִ����ת
    cv::Mat rotated_image;
    cv::warpAffine(image, rotated_image, rotation_mat, cv::Size(bound_w, bound_h));
    return rotated_image;
}
