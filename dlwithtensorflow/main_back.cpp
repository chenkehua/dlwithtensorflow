#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"text_det.h"
#include"table_det.h"
#include"text_angle_cls.h"
#include"text_rec.h"
#include"utility.h"
using namespace cv;
using namespace std;
using namespace Ort;

std::vector<std::vector<std::vector<int>>> convertToVectorOfVectorOfVectors(const std::vector<std::vector<cv::Point2f>>& points) {
    std::vector<std::vector<std::vector<int>>> result;
    for (const auto& outer_vector : points) {
        std::vector<std::vector<int>> inner_vector;
        for (const auto& point : outer_vector) {
            inner_vector.emplace_back(std::vector<int>{static_cast<int>(point.x), static_cast<int>(point.y)});
        }
        result.emplace_back(inner_vector);
    }
    return result;
}

static bool comparison_box(const OCR::OCRPredictResult &result1,
                         const OCR::OCRPredictResult &result2) {
    if (result1.box[0][1] < result2.box[0][1]) {
        return true;
    } else if (result1.box[0][1] == result2.box[0][1]) {
        return result1.box[0][0] < result2.box[0][0];
    } else {
        return false;
    }
}

void sorted_boxes(std::vector<OCR::OCRPredictResult> &ocr_result) {
  std::sort(ocr_result.begin(), ocr_result.end(), comparison_box);
  if (ocr_result.size() > 0) {
    for (int i = 0; i < ocr_result.size() - 1; i++) {
      for (int j = i; j >= 0; j--) {
        if (abs(ocr_result[j + 1].box[0][1] - ocr_result[j].box[0][1]) < 10 &&
            (ocr_result[j + 1].box[0][0] < ocr_result[j].box[0][0])) {
          std::swap(ocr_result[i], ocr_result[i + 1]);
        }
      }
    }
  }
}

cv::Mat crop_image(cv::Mat &img, const std::vector<int> &box) {
  cv::Mat crop_im;
  int crop_x1 = std::max(0, box[0]);
  int crop_y1 = std::max(0, box[1]);
  int crop_x2 = std::min(img.cols - 1, box[2] - 1);
  int crop_y2 = std::min(img.rows - 1, box[3] - 1);

  crop_im = cv::Mat::zeros(box[3] - box[1], box[2] - box[0], 16);
  cv::Mat crop_im_window =
      crop_im(cv::Range(crop_y1 - box[1], crop_y2 + 1 - box[1]),
              cv::Range(crop_x1 - box[0], crop_x2 + 1 - box[0]));
  cv::Mat roi_img =
      img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
  crop_im_window += roi_img;
  return crop_im;
}

void VisualizeBboxes(const cv::Mat &srcimg,
                              const OCR::StructurePredictResult &structure_result,
                              const std::string &save_path) {
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  //img_vis = crop_image(img_vis, structure_result.box);
  for (int n = 0; n < structure_result.cell_box.size(); n++) {
    if (structure_result.cell_box[n].size() == 8) {
      cv::Point rook_points[4];
      for (int m = 0; m < structure_result.cell_box[n].size(); m += 2) {
        rook_points[m / 2] =
            cv::Point(int(structure_result.cell_box[n][m]),
                      int(structure_result.cell_box[n][m + 1]));
      }
      const cv::Point *ppt[1] = {rook_points};
      int npt[] = {4};
      cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    } else if (structure_result.cell_box[n].size() == 4) {
      cv::Point rook_points[2];
      rook_points[0] = cv::Point(int(structure_result.cell_box[n][0]),
                                 int(structure_result.cell_box[n][1]));
      rook_points[1] = cv::Point(int(structure_result.cell_box[n][2]),
                                 int(structure_result.cell_box[n][3]));
      cv::rectangle(img_vis, rook_points[0], rook_points[1], CV_RGB(0, 255, 0),
                    2, 8, 0);
    }
  }

  cv::imwrite(save_path, img_vis);
  std::cout << "The table visualized image saved in " + save_path << std::endl;
}

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

    std::vector<OCR::OCRPredictResult> ocr_results;
	vector< vector<Point2f> > results = detect_model.detect(srcimg);
    std::vector<std::vector<std::vector<int>>> results_detect = convertToVectorOfVectorOfVectors(results);
    for (int i = 0; i < results_detect.size(); i++) {
        OCR::OCRPredictResult res;
        res.box = results_detect[i];
        ocr_results.push_back(res);
    }
    //sorted_boxes(ocr_results);
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
    for (int i =0 ; i < img_error.size() ;i++){
        ocr_results.erase(ocr_results.begin() + img_error[i]);
    }
    
    // miss bug
    
    for(int i = 0; i < img_list.size(); i++){
        ocr_results[i].text = rec_texts[i];
        ocr_results[i].score = rec_text_scores[i];           
    }

    std::string html;
    std::cout << "++++++++structure_html_tags+++++++++++++++++++++++++++++++++++++++++++++" <<std::endl;
    std::cout << "structure_html_tags: " << structure_html_tags[0].size() << std::endl;
    std::cout << "ocr_results: " << ocr_results.size() << std::endl;
    std::cout << "++++++++structure_html_tags+++++++++++++++++++++++++++++++++++++++++++++" <<std::endl;
    for(int i = 0; i < structure_html_tags.size(); i++){
        for(int j = 0; j < structure_html_tags[0].size(); j++){
            std::cout << structure_html_tags[i][j] << " " ;
        }
    }
    std::cout << "++++++++structure_html_tags+++++++++++++++++++++++++++++++++++++++++++++" <<std::endl;
    std::cout << "++++++++,structure_boxes+++++++++++++++++++++++++++++++++++++++++++++" <<std::endl;
    std::cout << "--------- check " << structure_boxes[0].size() << std::endl;
    std::cout << "--------- check " << structure_boxes[0][0].size() << std::endl;
    std::cout << "++++++++,structure_boxes+++++++++++++++++++++++++++++++++++++++++++++" <<std::endl;
    for(int i = 0; i < structure_boxes[0].size(); i++){
        for(int j = 0; j < structure_boxes[0][0].size(); j++){
            std::cout << structure_boxes[0][i][j] << " " ;
        }
    }
    std::cout << "++++++++,structure_boxes+++++++++++++++++++++++++++++++++++++++++++++" <<std::endl;

    html = table_model.rebuild_table(structure_html_tags[0],structure_boxes[0],ocr_results);
    
    std::cout << html << std::endl;
    std::vector<OCR::StructurePredictResult> structure_results;
    OCR::StructurePredictResult res;
    res.type = "table";
    res.box = std::vector<float>(4, 0.0);
    res.box[2] = srcimg.cols;
    res.box[3] = srcimg.rows;
    structure_results.push_back(res);
    structure_results[0].html = html;
    structure_results[0].cell_box = structure_boxes[0];
    structure_results[0].html_score = structure_scores[0];
    VisualizeBboxes(srcimg, structure_results[0],
                                   "out/" + std::to_string(0) +
                                       "_table.jpg");
}
