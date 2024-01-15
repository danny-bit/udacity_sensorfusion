#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;

    string descriptorClass;
    if (descriptorType.compare("SIFT") == 0) {
        descriptorClass = "DES_HOG";
    }
    else {
        descriptorClass = "DES_BINARY";
    }     

    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
		// Brute Force Matcher
        int normType;
        if (descriptorClass.compare("DES_BINARY") == 0)
            normType = cv::NORM_HAMMING;
        else
            normType = cv::NORM_L2;

        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    { 
		// Fast Library for Approximate Nearest Neighbors
		if (descSource.type() != CV_32F) 
		{
            // opencv bug
			descSource.convertTo(descSource, CV_32F);
			descRef.convertTo(descRef, CV_32F);
		}

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { 
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors 
        double minDescDistRatio = 0.8;
        int nMatches = 2;

        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, nMatches);

        // MP.6: descriptor distance ratio filtering 
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        // Binary Robust Invariant Scalable Keypoints
        // (binary; fixed sampling pattern)

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        // Binary Robust Indepenent Elementary Features
        // (binary)
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        // Oriented FAST and Rotated BRIEF
        // (binary)
        int nFeatures = 200;
        float scaleFactor = 1.0f;
        int nLevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;

        extractor = cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, 
                                    firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        // Fast retina keypoint.
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        const std::vector<int> selectedPairs = std::vector<int>();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, 
                                                   patternScale, nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        // A-KAZE (Accelerated-KAZE features)
        // (binary; M-LDB: modified-local difference binary descriptor)

        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorSize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
        extractor = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, 
                                      threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        // Scale Invariant Feature Transform
        // (gradient based)

        int nFeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10.0;
        double sigma = 1.6;

        extractor = cv::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

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

    bool zToFile = true;

    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if (zToFile)
        imwrite(windowName, visImage);
    else
    {
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}