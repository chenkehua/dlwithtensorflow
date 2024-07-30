#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "utility.h"
using namespace cv;
using namespace std;
using namespace Ort;

class TableDetector
{
public:
	TableDetector();
	//vector< vector<Point2f> > detect(Mat& srcimg);
	void detect(Mat& srcimg,std::vector<std::vector<std::string>> &structure_html_tags,std::vector<float> &structure_scores,std::vector<std::vector<std::vector<int>>> &structure_boxes);
	void draw_pred(Mat& srcimg, vector< vector<Point2f> > results);
	Mat get_rotate_crop_image(const Mat& frame, vector<Point2f> vertices);

    std::string rebuild_table(std::vector<std::string> structure_html_tags,std::vector<std::vector<int>> structure_boxes,std::vector<OCR::OCRPredictResult> &ocr_result);
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
	//void normalize_(Mat img);
    void padding(const cv::Mat &img, cv::Mat &resize_img,const int max_len);
	void normalize_(Mat *im);

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
