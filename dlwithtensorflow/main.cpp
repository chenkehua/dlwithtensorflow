#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"text_det.h"
#include"layout_det.h"
#include"table_det.h"
#include"text_angle_cls.h"
#include"text_rec.h"
#include"utility.h"
//#include "elemutil.h"

using namespace cv;
using namespace std;
using namespace Ort;


int main()
{
	TextDetector detect_model;
	TableDetector table_model;
	TextRecognizer rec_model;
	
    string imgpath = "table.jpg";
	Mat srcimg = imread(imgpath);
    std::vector<std::vector<std::string>> structure_html_tags;
    std::vector<float> structure_scores(1, 0);
    std::vector<std::vector<std::vector<int>>> structure_boxes;
	table_model.detect(srcimg,structure_html_tags,structure_scores,structure_boxes);
    /*
	TextDetector detect_model;
	LayoutDetector layout_model;
	//TextClassifier angle_model;
	TextRecognizer rec_model;

	string imgpath = "1.png";
	Mat srcimg = imread(imgpath);
	///cv::rotate(srcimg, srcimg, 1);

	//vector< vector<Point2f> > results = detect_model.detect(srcimg);
	std::vector<OCR::StructurePredictResult> structure_results = layout_model.detect(srcimg);
    std::vector<int> bbox_num;
    bbox_num.push_back(structure_results.size());
    cv::Scalar color(0,255,0);
    for (int j =0; j < structure_results.size(); j++){
          cv::rectangle(srcimg, cv::Point(structure_results[j].box[0], structure_results[j].box[1]), cv::Point(structure_results[j].box[2], structure_results[j].box[3]), color ,2);
          cv::Point loc(structure_results[j].box[0], structure_results[j].box[1]-10);
          cv::putText(srcimg,structure_results[j].type ,loc, cv::FONT_HERSHEY_SIMPLEX,0.5,color,2);

    }
    cv::imwrite("newlay.jpg",srcimg);
    */
	/*
	for (size_t i = 0; i < results_layout.size(); i++){
		for (int j = 0; j < 4; j++){
			std::cout << "x: "<< results_layout[i][j].x << " y: " << results_layout[i][j].y << std::endl;
		}
	}*/
	/*
	for (size_t i = 0; i < results.size(); i++){
		Mat textimg = detect_model.get_rotate_crop_image(srcimg, results[i]);
		string text = rec_model.predict_text(textimg);
		cout << text << endl;
	}*/
	//detect_model.draw_pred(srcimg, results);
	//imwrite("result.jpg", srcimg);
}
