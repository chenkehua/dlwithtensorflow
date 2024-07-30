#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"text_det.h"
#include"text_rec.h"
#include"utility.h"
using namespace cv;
using namespace std;
using namespace Ort;

int main()
{
	TextDetector detect_model;
	TextRecognizer rec_model;
	string imgpath = "table.jpg";
	Mat srcimg = imread(imgpath);
    
    std::vector<OCR::OCRPredictResult> ocr_results;
	vector< vector<Point2f> > results = detect_model.detect(srcimg);
    std::vector<Mat> img_list;
    std::vector<int> img_error;

	for (size_t i = 0; i < results.size(); i++)
	{
		Mat textimg = detect_model.get_rotate_crop_image(srcimg, results[i]);
        if (textimg.empty()) {
            img_error.push_back(i);
            //img_list.push_back(Mat());
            continue;
        }
        
        img_list.push_back(textimg);
		//string text = rec_model.predict_text(textimg);
		//cout << text << endl;
	}

    std::vector<std::string> rec_texts(img_list.size(), "");
    std::vector<float> rec_text_scores(img_list.size(), 0);

	rec_model.batch_predict_text(img_list,rec_texts,rec_text_scores);
    //for (int i =0 ; i < img_error.size() ;i++){
    //    ocr_results.erase(ocr_results.begin() + img_error[i]);
    //}
}
