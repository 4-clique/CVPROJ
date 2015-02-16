#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

Mat src; Mat dst;
char window_name1[] = "Unprocessed Image";
char window_name2[] = "Processed Image";
char key = 'a';
bool equalizeHistogram = false;

int main( int argc, char** argv )
{
    Mat cover = imread("lego.jpg");
    if(!cover.data) return -1;

    //Threshhold para a qualidade de features
    int minHessian = 400;

    //Criação do detector e extração dos pontos chaves do cover
    SurfFeatureDetector detector (minHessian);
    vector<KeyPoint> keysCover;
    detector.detect(cover,keysCover);

    //Calculando o descritor do cover
    SurfDescriptorExtractor extractor;
    Mat descritor;
    extractor.compute(cover, keysCover, descritor);


    //Mat drawKeys;
    //drawKeypoints(cover, keysCover, drawKeys, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("KeyPoint",drawKeys);

    VideoCapture capture(0);

    while(key != 'z'){

        //Captura de frame
        Mat frame;
        capture >> frame;

        if(frame.data){
            Mat img;
            img = frame;
            
            //Equalização de histograma para melhorar rastramento
            if(equalizeHistogram){
                vector<Mat> channels;
                Mat hist_img;
                cvtColor(img,hist_img,CV_BGR2YCrCb);
                split(hist_img, channels);
                equalizeHist(channels[0],channels[0]);
                merge(channels,hist_img);
                cvtColor(hist_img,img,CV_YCrCb2BGR);
            }

            //Criação do detector e extração dos pontos chaves do frame
            vector<KeyPoint> keysFrame;
            detector.detect(img,keysFrame);

            //Calculando o descritor do frame
            Mat descritorFrame;
            extractor.compute(img, keysFrame, descritorFrame);


            //Matching com força bruta
            BFMatcher matcher;
            vector<DMatch> matches;
            matcher.match(descritor,descritorFrame,matches);


            //Tentamos apenas capturar os melhores matches
            double min_dis = 100, max_dis = 0;
            for( int i = 0; i < descritor.rows; i++ ){
                if( matches[i].distance < min_dis ) min_dis = matches[i].distance;
                if( matches[i].distance > max_dis ) max_dis = matches[i].distance;
            }
            vector<DMatch> good_matches;
            for( int i = 0; i < descritor.rows; i++ ){
                if( matches[i].distance < 2.5*min_dis )
                    good_matches.push_back( matches[i]);
            }

            //Desenhar matches
            Mat img_matches;
            drawMatches(cover, keysCover, img, keysFrame, good_matches, img_matches,
                Scalar::all(-1),Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            //Localizar objetos
            vector<Point2f> obj;
            vector<Point2f> scene;

            for( int i = 0; i < good_matches.size(); i++ ){
                obj.push_back( keysCover[ good_matches[i].queryIdx ].pt );
                scene.push_back( keysFrame[ good_matches[i].trainIdx ].pt );
            }

            if(obj.size() >= 4 && scene.size() >= 4){
                //Localizando a homografia
                Mat H = findHomography( obj, scene, CV_RANSAC );

                //Pegando as quina
                std::vector<Point2f> obj_corners(4);
                obj_corners[0] = cvPoint(0,0); 
                obj_corners[1] = cvPoint( cover.cols, 0 );
                obj_corners[2] = cvPoint( cover.cols, cover.rows );
                obj_corners[3] = cvPoint( 0, cover.rows );
                std::vector<Point2f> scene_corners(4);

                //Mudando a perspectiva para de mundo
                perspectiveTransform( obj_corners, scene_corners, H);

                line( img_matches, scene_corners[0] + Point2f( cover.cols, 0), scene_corners[1] + Point2f( cover.cols, 0), Scalar(0, 255, 0), 4 );
                line( img_matches, scene_corners[1] + Point2f( cover.cols, 0), scene_corners[2] + Point2f( cover.cols, 0), Scalar( 0, 255, 0), 4 );
                line( img_matches, scene_corners[2] + Point2f( cover.cols, 0), scene_corners[3] + Point2f( cover.cols, 0), Scalar( 0, 255, 0), 4 );
                line( img_matches, scene_corners[3] + Point2f( cover.cols, 0), scene_corners[0] + Point2f( cover.cols, 0), Scalar( 0, 255, 0), 4 );


            }
            imshow("Matches", img_matches);
            

            /**if(!drawKeys.data) return -1;
            else
            imshow("KeyPoint 1", drawKeys);**/
        }
        
        key = waitKey(1);
        if(key == 'e') equalizeHistogram = !equalizeHistogram;
    }
    
}