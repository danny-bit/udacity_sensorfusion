
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, int imgIndex)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m, \nyw_min=%2.2f", xwmin, ywmax-ywmin, ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    //cv::namedWindow(windowName, 1);
    //cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }

    cv::imwrite("top"+std::to_string(imgIndex)+".jpg", topviewImg);
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // boundingBox: currFrame
    // kptMatches: matches associated with current/prev frame (trainIdx=>currFrame, queryIdx=>prevFrame)

    for (auto match : kptMatches)
    { // loop through kptMatches
        cv::KeyPoint kptCurrent = kptsCurr[match.trainIdx];

        if (boundingBox.roi.contains(kptCurrent.pt))
        { // current match is within bounding box
            boundingBox.keypoints.push_back(kptCurrent);
            boundingBox.kptMatches.push_back(match);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // kptsPrev: all keypoints of prev frame
    // kptsCurr: all keypoints of curr frame
    // kptMatches: matches associated with specific bounding box (trainIdx=>currFrame, queryIdx=>prevFrame)

    std::vector<double> vecDistRatios;
    double minDist = 90.0;
    
    for (auto match : kptMatches)
    {
        cv::KeyPoint kptOuterPrev = kptsPrev[match.queryIdx];
        cv::KeyPoint kptOuterCurr = kptsCurr[match.trainIdx];
        
        for (auto matchInner : kptMatches)
        {
            cv::KeyPoint kptInnerPrev = kptsPrev[matchInner.queryIdx];
            cv::KeyPoint kptInnerCurr = kptsCurr[matchInner.trainIdx];

            double distCurr = cv::norm(kptOuterCurr.pt - kptInnerCurr.pt);
            double distPrev = cv::norm(kptOuterPrev.pt - kptInnerPrev.pt);

            if (distPrev > 1e-8 && distCurr > minDist)
            {
                double distRatio = distCurr / distPrev;
                vecDistRatios.push_back(distRatio);
            }
        }
    }

    auto const nRatios = static_cast<float>(vecDistRatios.size());

    bool printDebug = false;   
    if (printDebug)
    {
        for (float print : vecDistRatios)
            std::cout << print << ", ";
        std::cout << endl;
    }

    double dT = 1.0 / frameRate;

    // mean
    //double meanDistRatio = std::accumulate(vecDistRatios.begin(), vecDistRatios.end(), 0.0) / nRatios;
    // median
    size_t na = vecDistRatios.size() / 2;
    std::nth_element(vecDistRatios.begin(), vecDistRatios.begin() + na, vecDistRatios.end());
    double medianDistRatio = vecDistRatios[na];

    TTC = (-1.0/ (1 - medianDistRatio))*dT;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    std::vector<float> prevDistVec, curDistVec;
    float prevMin=0.0;
    float prevMax=0.0;
    float prevMean=0.0;
    float curMin=0.0;
    float curMax=0.0;
    float curMean=0.0;

    for (auto it: lidarPointsPrev)
    {
        prevMin = prevMin<it.y ? prevMin : it.y;
        prevMax = prevMax>it.y ? prevMax : it.y;
    }

    prevMean = (prevMin+prevMax)/2;

    for (auto it: lidarPointsCurr)
    {
        curMin = curMin<it.y ? curMin : it.y;
        curMax = curMax>it.y ? curMax : it.y;
    }

    curMean = (curMin+curMax)/2;

    for (auto it: lidarPointsPrev)
        if (abs(it.y-prevMean) <= 0.5)
            prevDistVec.push_back(it.x);

    for (auto it: lidarPointsCurr)
        if (abs(it.y-curMean) <= 0.5)
            curDistVec.push_back(it.x);

    size_t na = prevDistVec.size() * 0.6;
    std::nth_element(prevDistVec.begin(), prevDistVec.begin() + na, prevDistVec.end());
    size_t nb = curDistVec.size() * 0.6;
    std::nth_element(curDistVec.begin(), curDistVec.begin() + nb, curDistVec.end());

    double prevDist = prevDistVec[na];
    double curDist  = curDistVec[nb];

    /*size_t na = prevDistVec.size() / 2;
    std::nth_element(prevDistVec.begin(), prevDistVec.begin() + na, prevDistVec.end());
    size_t nb = curDistVec.size() / 2;
    std::nth_element(curDistVec.begin(), curDistVec.begin() + nb, curDistVec.end());

    double prevDist = prevDistVec[na];
    double curDist  = curDistVec[nb];*/

    std::cout << "prev" << std::to_string(prevDist) << ",curr " << std::to_string(curDist) << std::endl;

    double dT = 1.0 / frameRate;
    TTC = curDist/(prevDist-curDist) * dT;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // bbBestMatches: map[currentFrameBoxID] -> prevFrameBoxID
    // note1: only a single in the previous frame is allowed
    // note2: match is done by average keypoint distance

    int prevMaxBoxID;
    for (auto it_box : prevFrame.boundingBoxes)
        prevMaxBoxID = max(prevMaxBoxID, it_box.boxID);

    int currMaxBoxID;
    for (auto it_box : currFrame.boundingBoxes)
        currMaxBoxID = max(currMaxBoxID, it_box.boxID);

    int minNumMatchedKeypoints = 4;
    
    // row: curr boxID, col: prev boxID
    float initMax = 1e5; // initialize with high value: dirty trick to detect valid minimum
    std::vector<std::vector<float>> matBoxSimilarity(currMaxBoxID+1, 
                                                    std::vector<float> (prevMaxBoxID+1, initMax));
    std::vector<std::vector<int>> matBoxCount(currMaxBoxID+1, 
                                              std::vector<int> (prevMaxBoxID+1));

    for (auto it_box : prevFrame.boundingBoxes)
    {
        for (auto match: matches) 
        { 
            cv::KeyPoint* trainKpt = &prevFrame.keypoints[match.queryIdx];
            if (it_box.roi.contains(trainKpt->pt))
            { // keypoint match is within prev frame bounding box,
              // search corresponding current frame bounding box 

                cv::KeyPoint queryKpt = currFrame.keypoints[match.trainIdx];
                for (auto it_match_box : currFrame.boundingBoxes)
                { 
                    if (it_match_box.roi.contains(queryKpt.pt))
                    { // corresponding box found, add keypoint distance :: alternative - count?
                        matBoxSimilarity[it_match_box.boxID][it_box.boxID] += match.distance;
                        matBoxCount[it_match_box.boxID][it_box.boxID] += 1;
                    }
                }
            }
        }
    }

    // normalize similarity measure matrix / suppress matches with few keypoints
    for (int idxRow = 0; idxRow <= currMaxBoxID; idxRow++)
        for (int idxCol = 0; idxCol <= prevMaxBoxID; idxCol++)
            if (matBoxCount[idxRow][idxCol] < minNumMatchedKeypoints)
                matBoxSimilarity[idxRow][idxCol] = initMax;
            else
                matBoxSimilarity[idxRow][idxCol] = matBoxSimilarity[idxRow][idxCol] / (matBoxCount[idxRow][idxCol]);


    bool print_debug  = false;

    if (print_debug)
    {
        for (int idxBoxID = 0; idxBoxID <= currMaxBoxID; idxBoxID++)
        {
            for (float i: matBoxSimilarity[idxBoxID])
                std::cout << i << ' ';

            std::cout << std::endl;
        }  
    }

    for (int idxBoxID = 0; idxBoxID <= currMaxBoxID; idxBoxID++)
    {
        auto it = std::min_element(matBoxSimilarity[idxBoxID].begin(), 
                                   matBoxSimilarity[idxBoxID].end());
        
        if (*it < initMax)
        { // min found
            int prev_idxBestMatch = std::distance(matBoxSimilarity[idxBoxID].begin(), it);
            bbBestMatches[idxBoxID] = prev_idxBestMatch; // current -> prev
        
            // prevent second match?
            for (int idx = 0; idx <= currMaxBoxID; idx++)
            {
                matBoxSimilarity[idx][prev_idxBestMatch] = initMax;
            }
        }
    }

    if (print_debug)
    {
        for (auto const& x : bbBestMatches)
        {
        std::cout << x.first  // string (key)
                << ':' 
                << x.second // string's value 
                << std::endl;
        }
    }
}