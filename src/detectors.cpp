#include "detectors.hpp"

using namespace std;

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = static_cast<int>(img.rows * img.cols / max(1.0, minDistance)); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = static_cast<float>(blockSize);
        keypoints.push_back(newKeyPoint);
    }
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

//// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT 
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;   // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;     // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8

        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::SIFT::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }

    detector->detect(img,keypoints);
}

void detKeypointsHarris (vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 3;       // pixel neighborhood for detection
    int apertureSize = 3;    // sobel operator parameter [must be ODD]
    double k = 0.04;         // harris magic const.
    double threshold = 100;  // corner threshold
    double max_overlap = 0.0; // overlap in %

    cv::Mat dst = cv::Mat::zeros(img.size(), img.type());
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (size_t idx_row = 0; idx_row < dst_norm.rows; idx_row++)
    {
        for (size_t idx_col = 0; idx_col < dst_norm.cols; idx_col++)
        {
            int px_response = static_cast<int>(dst_norm.at<float>(idx_row, idx_col));
            if (px_response > threshold)
            {
                // ## case: keypoint found
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(idx_col, idx_row);
                keypoint.size = 2*apertureSize; 
                keypoint.response = px_response;

                // NMS (non-maximum suppression) 
                bool is_overlap = false;
                for (auto it = keypoints.begin(); it !=keypoints.end(); ++it) 
                {
                    double area_overlap = cv::KeyPoint::overlap(keypoint, *it);
                    if (area_overlap > max_overlap)
                    {
                        is_overlap = true;
                        if (keypoint.response > (*it).response)
                        {
                           *it = keypoint; // replace the keypoint
                           break;
                        }
                    }
                }

                if (!is_overlap)
                {
                    keypoints.push_back(keypoint);
                }
            } // eif: keypoint found
        } // eol: columns
    } // eol: rows

    // visualize results
    if (bVis)
    {
        doVisKeypoints(keypoints, img, "Harris Corner Detector Results");
    }
}

void doVisKeypoints (std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, string windowName)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}