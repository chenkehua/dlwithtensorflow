#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "utility.h"
using namespace cv;
using namespace std;
using namespace Ort;

class LayoutDetector
{
public:
	LayoutDetector();
	//vector< vector<Point2f> > detect(Mat& srcimg);
	std::vector<OCR::StructurePredictResult> detect(Mat& srcimg);
	//std::vector<StructurePredictResult> detect(Mat& srcimg);
	void draw_pred(Mat& srcimg, vector< vector<Point2f> > results);
	Mat get_rotate_crop_image(const Mat& frame, vector<Point2f> vertices);
    //std::vector<float> activation_function_softmax(std::vector<float> &src);
private:
	float binaryThreshold;
	float polygonThreshold;
	float unclipRatio;
	int maxCandidates;
	const int longSideThresh = 3;
	const int short_size = 736;
	const float meanValues[3] = { 0.485, 0.456, 0.406 };
	const float normValues[3] = { 0.229, 0.224, 0.225 };
	float contourScore(const Mat& binary, const vector<Point>& contour);
	void unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly);
	vector< vector<Point2f> > order_points_clockwise(vector< vector<Point2f> > results);
    //float fast_exp(float x);

	Mat preprocess(Mat srcimg);
	vector<float> input_image_;
	void normalize_(Mat img);

    double score_threshold_ = 0.4;
    double nms_threshold = 0.5;
    std::vector<int> fpn_stride_ = {8, 16, 32, 64};
    int num_class_ = 5;
	Session *net;
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Picodet");
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
    std::vector<OCR::StructurePredictResult> result;
    //std::vector<StructurePredictResult> result;
};
