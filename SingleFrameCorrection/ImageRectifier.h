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

// ͼ��������࣬��װ����ͼ�������ع���
class ImageRectifier {
private:
    // ��������
    const std::string WGS84 = "EPSG:4326";
    const std::string CGCS2000 = "EPSG:4490";

    // �������
    double pixel_size_;    // ���ش�С���ף�
    double focal_length_;  // ���ࣨ�ף�

    // �ⷽλԪ�� (x, y, z, omega, phi, kappa)
    Eigen::Matrix<double, 6, 1> eo_;

    // ͼ�����
    cv::Mat image_;
    cv::Mat restored_image_;

    // ����ϵͳ���
    int epsg_code_;
    std::string target_crs_;

    // ��ת����
    Eigen::Matrix3d rotation_matrix_;

public:
    /**
     * ���캯��
     * @param pixel_size ���ش�С���ף�
     * @param focal_length ���ࣨ�ף�
     */
    ImageRectifier(double pixel_size, double focal_length);

    /**
     * ��������
     */
    ~ImageRectifier();

    /**
     * ����ͼ��
     * @param file_path ͼ���ļ�·��
     * @return �Ƿ���سɹ�
     */
    bool loadImage(const std::string& file_path);

    /**
     * �����ⷽλԪ��
     * @param longitude ���ȣ��ȣ�
     * @param latitude γ�ȣ��ȣ�
     * @param height �߶ȣ��ף�
     * @param roll ��ת�ǣ��ȣ�
     * @param pitch �����ǣ��ȣ�
     * @param yaw ƫ���ǣ��ȣ�
     * @param coordinate_system ����ϵ��Ĭ��WGS84��
     */
    void setExteriorOrientation(double longitude, double latitude, double height,
        double roll, double pitch, double yaw,
        const std::string& coordinate_system = "EPSG:4326");

    /**
     * �ָ�ͼ����
     * @param orientation �������
     */
    void restoreImageOrientation(int orientation = 1);

    /**
     * ����ͼ��߽��
     * @param dem ���ָ߳�ģ��ֵ
     * @return �߽�����
     */
    Eigen::Matrix<double, 4, 1> calculateBoundary(double dem = 0);

    /**
     * ִ��ƽ��ƽ�о���
     * @param boundary �߽��
     * @param gsd �����������
     * @param ground_height ����߶�
     * @return �������B��G��R��Alphaͨ��
     */
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> rectifyPlaneParallel(
        const Eigen::Matrix<double, 4, 1>& boundary, double gsd, double ground_height = 0);

    /**
     * �������е���ο���TIFF�ļ�
     * @param b ��ɫͨ��
     * @param g ��ɫͨ��
     * @param r ��ɫͨ��
     * @param a Alphaͨ��
     * @param boundary �߽��
     * @param gsd �����������
     * @param rows ͼ��߶�
     * @param cols ͼ����
     * @param dst ����ļ�·����������չ����
     */
    void createGeoTiff(const cv::Mat& b, const cv::Mat& g, const cv::Mat& r, const cv::Mat& a,
        const Eigen::Matrix<double, 4, 1>& boundary, double gsd,
        int rows, int cols, const std::string& dst);

    /**
     * ������ͼ���������
     * @param input_path ����ͼ��·��
     * @param output_dir ���Ŀ¼
     * @param dem ���ָ߳�ģ��ֵ
     * @param orientation ͼ����
     */
    void processImage(const std::string& input_path, const std::string& output_dir,
        double dem = 0, int orientation = 1);

private:
    /**
     * ����EPSG����
     * @param longitude ����
     * @param latitude γ��
     * @param coordinate_system ����ϵ
     * @return EPSG����
     */
    int epsg_calc(double longitude, double latitude, const std::string& coordinate_system);

    /**
     * ��������תƽ������
     * @param lon ����
     * @param lat γ��
     * @param target_crs Ŀ������ϵ
     * @param source_crs Դ����ϵ
     * @return ƽ������
     */
    std::vector<double> geographic2plane(double lon, double lat,
        const std::string& target_crs,
        const std::string& source_crs);

    /**
     * RPYתOPK
     * @param rpy RPY��
     * @param maker �豸������
     * @return OPK��
     */
    std::vector<double> rpy_to_opk(const std::vector<double>& rpy, const std::string& maker = "");

    /**
     * �Ƕ�ת����
     * @param degrees �Ƕ�ֵ
     * @return ����ֵ
     */
    double degrees_to_radians(double degrees);

    /**
     * 2D��ת����
     * @param theta �Ƕȣ����ȣ�
     * @return 2D��ת����
     */
    Eigen::Matrix2d rot2d(double theta);

    /**
     * 3D��ת����
     * @param omega ŷ��٤�ǣ����ȣ�
     * @param phi �ƽǣ����ȣ�
     * @param kappa �����ǣ����ȣ�
     * @return 3D��ת����
     */
    Eigen::Matrix3d Rot3D(double omega, double phi, double kappa);

    /**
     * ��ȡͼ�񶥵�
     * @return �������
     */
    Eigen::Matrix<double, 3, 4> getVertices();

    /**
     * ͶӰ����
     * @param vertices �������
     * @param dem ���ָ߳�ģ��ֵ
     * @return ͶӰ����
     */
    Eigen::Matrix<double, 2, 4> projection(const Eigen::Matrix<double, 3, 4>& vertices, double dem);

    /**
     * ��תͼ��
     * @param image ����ͼ��
     * @param angle ��ת�Ƕ�
     * @return ��ת���ͼ��
     */
    cv::Mat rotateImage(const cv::Mat& image, double angle);
};

#endif // IMAGE_RECTIFIER_H
