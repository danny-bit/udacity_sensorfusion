
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;  // "speed"
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    std::ofstream logfile;
    logfile.open("log.csv", std::ios_base::out); // append instead of overwrite
    bool bIsFirstLine = true;
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        DataFrame* currFrame = &(*(dataBuffer.end()-1));

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects(currFrame->cameraImg, currFrame->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9; // assumption: level road surface
        float minX =  2.0, maxX = 20.0; // assumption: only look 20m apart
        float maxY = 2.0; // assumption: focus on ego lane (2m left / 2m right)
        float minR = 0.1; // assumption: suppress small responses

        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        currFrame->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI(currFrame->boundingBoxes, currFrame->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = true;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1000, 1000), false, imgIndex);
        }
        bVis = false;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = "SHITOMASI";
        string detectorType = "SIFT";

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else
        {
            //// FAST, BRISK, ORB, AKAZE
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        doVisKeypoints(keypoints, imgGray, "keypts"+std::to_string(imgIndex)+".png");

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        currFrame->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        //string descriptorType = "ORB"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        string descriptorType = "SIFT"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints(currFrame->keypoints, currFrame->cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        currFrame->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            DataFrame* prevFrame = &(*(dataBuffer.end()-2));

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            matchDescriptors(prevFrame->keypoints, currFrame->keypoints,
                             prevFrame->descriptors, currFrame->descriptors,
                             matches, descriptorType, matcherType, selectorType);


            // store matches in current data frame
            currFrame->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

            //continue; // skips directly to the next image without processing what comes beneath

            bool bVisMatches = true;
            bool bWait = false;
            if (bVisMatches)
            {
                // draw all of the current frame boxes and correspondance findings from previous frame boxes in same color
                // bbBest: prev->current
                cv::Mat matchImg = (currFrame->cameraImg).clone();
                // prev: left; current right

                //DataFrame* prevImg = prevFrame;
                /*cv::drawMatches(prevFrame->cameraImg, prevFrame->keypoints,
                                currFrame->cameraImg, currFrame->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);*/
                cv::hconcat(currFrame->cameraImg, 
                    	    prevFrame->cameraImg, matchImg);

                std::vector<cv::Scalar> colorMap = {
                                                    cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(255,0,0),
                                                    cv::Scalar(0,255,255), cv::Scalar(255,255,0), cv::Scalar(255,0,255),
                                                    cv::Scalar(0,0,100), cv::Scalar(0,100,0), cv::Scalar(100,0,0),
                                                    cv::Scalar(0,100,100), cv::Scalar(100,100,0), cv::Scalar(100,0,100)
                                                  };

                //cv::Scalar textColor = cv::Scalar(255,255,255);
                cv::Scalar textColor = cv::Scalar(0,0,0);

                int idxColor = 0;
                for (auto currBB : currFrame->boundingBoxes)
                {
                    string label = std::to_string(currBB.boxID);

                    int x = currBB.roi.x;
                    int y = currBB.roi.y;
                    cv::rectangle(matchImg, cv::Point(x, y), 
                                            cv::Point(x + currBB.roi.width, y + currBB.roi.height), colorMap[idxColor], 2);


                    cv::Size txtSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
                    cv::rectangle(matchImg, cv::Point(x, y - 20), 
                                            cv::Point(x + txtSize.width, y), colorMap[idxColor], -1);
                    cv::putText(matchImg, label, 
                                cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);

                    if (bbBestMatches.count(currBB.boxID) > 0)
                    {
                        int relatedBoxId = bbBestMatches[currBB.boxID];

                        bool found = false;
                        BoundingBox relBB;
                        for (auto relBBCandidate : prevFrame->boundingBoxes)
                        {
                            if (relBBCandidate.boxID == relatedBoxId)
                            {
                                relBB = relBBCandidate;
                                found = true;
                                break;
                            }
                        }
                        if (found)
                        {
                            label.clear();
                            label = std::to_string(relBB.boxID);

                            int offset_x = prevFrame->cameraImg.size[1];
                            x = relBB.roi.x + offset_x;
                            y = relBB.roi.y;
                            cv::rectangle(matchImg, cv::Point(x, y), 
                                                    cv::Point(x +relBB.roi.width, y + relBB.roi.height), colorMap[idxColor], 2);

                            txtSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
                            cv::rectangle(matchImg, cv::Point(x, y - 20), 
                                                    cv::Point(x + txtSize.width, y), colorMap[idxColor], -1);
                            cv::putText(matchImg, label, 
                                        cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
                        }
                    }
                    idxColor++;
                    continue;
                }

                string windowName = "Matching keypoints between two camera images";
                //cv::namedWindow(windowName, 7);
                //cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;

                if (bWait)
                    cv::waitKey(0); // wait for key to be pressed
                cv::imwrite("match"+std::to_string(imgIndex)+".jpg",matchImg);
            }

            //continue; // skips directly to the next image without processing what comes beneath

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            // bbMatch [curr] = prev
            for (auto it_match = currFrame->bbMatches.begin(); it_match != currFrame->bbMatches.end(); ++it_match)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                bool prevBBfound = false;
                bool currBBfound = false;
                for (auto it2 = currFrame->boundingBoxes.begin(); it2 != currFrame->boundingBoxes.end(); ++it2)
                {
                    if (it_match->first== it2->boxID) 
                    {
                        currBBfound = true;
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = prevFrame->boundingBoxes.begin(); it2 != prevFrame->boundingBoxes.end(); ++it2)
                {
                    if (it_match->second == it2->boxID) 
                    {
                        prevBBfound=true;
                        prevBB = &(*it2);
                    }
                }

                if (!prevBBfound || !currBBfound)
				{
                    std::cout << "no matching bounding box" << std::endl;
                    continue;
				}

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, prevFrame->keypoints, currFrame->keypoints, currFrame->kptMatches);                    
                    computeTTCCamera(prevFrame->keypoints, currFrame->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    logfile << std::to_string(imgIndex) << ", ";
                    logfile << std::to_string(ttcLidar) << ", ";
                    logfile << std::to_string(ttcCamera) << std::endl;

                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = currFrame->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        //cv::namedWindow(windowName, 4);
                        //cv::imshow(windowName, visImg);
                        cv::imwrite("ttc"+std::to_string(imgIndex)+".jpg", visImg);
                        cout << "Press key to continue to next frame" << endl;
                        //cv::waitKey(0);

                        cv::Mat matchImg = (prevFrame->cameraImg).clone();
                        cv::drawMatches(prevFrame->cameraImg, prevFrame->keypoints,
                                        currFrame->cameraImg, currFrame->keypoints,
                                        currBB->kptMatches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        cv::imwrite("kptmatch_" + std::to_string(imgIndex) + ".png", matchImg);
                        cout << "Press key to continue to next image" << endl;
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images

    return 0;
}
