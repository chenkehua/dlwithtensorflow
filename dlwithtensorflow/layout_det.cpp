#include"layout_det.h"

float calculate_iou(std::vector<float> &box1, std::vector<float> &box2) {
  float area1 = std::max((float)0.0, box1[2] - box1[0]) *
                std::max((float)0.0, box1[3] - box1[1]);
  float area2 = std::max((float)0.0, box2[2] - box2[0]) *
                std::max((float)0.0, box2[3] - box2[1]);

  // computing the sum_area
  float sum_area = area1 + area2;

  // find the each point of intersect rectangle
  float x1 = std::max(box1[0], box2[0]);
  float y1 = std::max(box1[1], box2[1]);
  float x2 = std::min(box1[2], box2[2]);
  float y2 = std::min(box1[3], box2[3]);

  // judge if there is an intersect
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    float intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

void nms(std::vector<OCR::StructurePredictResult> &input_boxes,
                               float nms_threshold) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](OCR::StructurePredictResult a, OCR::StructurePredictResult b) {
              return a.confidence > b.confidence;
            });
  std::vector<int> picked(input_boxes.size(), 1);

  for (int i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 0) {
      continue;
    }
    for (int j = i + 1; j < input_boxes.size(); ++j) {
      if (picked[j] == 0) {
        continue;
      }
      float iou = calculate_iou(input_boxes[i].box, input_boxes[j].box);
      if (iou > nms_threshold) {
        picked[j] = 0;
      }
    }
  }
  std::vector<OCR::StructurePredictResult> input_boxes_nms;
  for (int i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 1) {
      input_boxes_nms.push_back(input_boxes[i]);
    }
  }
  input_boxes = input_boxes_nms;
}

float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}
std::vector<float> activation_function_softmax(std::vector<float> &src) {
  int length = src.size();
  std::vector<float> dst;
  dst.resize(length);
  const float alpha = float(*std::max_element(&src[0], &src[0 + length]));
  float denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return dst;
}

LayoutDetector::LayoutDetector()
{
	this->binaryThreshold = 0.3;
	this->polygonThreshold = 0.5;
	this->unclipRatio = 1.6;
	this->maxCandidates = 1000;
	string model_path = "inference/picodet_lcnet_x1_0_fgd_layout_infer/model.onnx";
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	net = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = net->GetInputCount();
	size_t numOutputNodes = net->GetOutputCount();
	std::cout << "numOutputNodes: " << numOutputNodes << std::endl;
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(net->GetInputName(i, allocator));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(net->GetOutputName(i, allocator));
	}
}

Mat LayoutDetector::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(608, 800));
	return dstimg;
}

void LayoutDetector::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - this->meanValues[c]) / this->normValues[c];
			}
		}
	}
}

OCR::StructurePredictResult disPred2Bbox(std::vector<float> bbox_pred, int label,
                                   float score, int x, int y, int stride,
                                   std::vector<int> im_shape, int reg_max) {
  std::vector<std::string> label_list_ = {"text","title","list","table","figure"};
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    std::vector<float> bbox_pred_i(bbox_pred.begin() + i * reg_max,
                                   bbox_pred.begin() + (i + 1) * reg_max);
    std::vector<float> dis_after_sm = activation_function_softmax(bbox_pred_i);
    for (int j = 0; j < reg_max; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
  }

  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)im_shape[1]);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)im_shape[0]);

  OCR::StructurePredictResult result_item;
  result_item.box = {xmin, ymin, xmax, ymax};
  result_item.type = label_list_[label];
  result_item.confidence = score;

  return result_item;
}

std::vector<OCR::StructurePredictResult> extract_result(std::vector<std::vector<float>> outs,
        std::vector<int> ori_shape,
        std::vector<int> resize_shape, int reg_max){
  int in_h = resize_shape[0];
  int in_w = resize_shape[1];
  float scale_factor_h = resize_shape[0] / float(ori_shape[0]);
  float scale_factor_w = resize_shape[1] / float(ori_shape[1]);
  std::vector<OCR::StructurePredictResult> results;
  std::vector<std::vector<OCR::StructurePredictResult>> bbox_results;
  int num_class_ = 5;
  bbox_results.resize(num_class_);
  double nms_threshold_ = 0.5;
  std::vector<int> fpn_stride_ = {8, 16, 32, 64};
  double score_threshold_ = 0.4;
  for (int i = 0; i < fpn_stride_.size(); ++i) {
    int feature_h = std::ceil((float)in_h / fpn_stride_[i]);
    int feature_w = std::ceil((float)in_w / fpn_stride_[i]);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      // score and label
      float score = 0;
      int cur_label = 0;
      for (int label = 0; label < num_class_; label++) {
        if (outs[i][idx * num_class_ + label] > score) {
          score = outs[i][idx * num_class_ + label];
          cur_label = label;
        }
      }
      // bbox
      if (score > score_threshold_) {
        int row = idx / feature_w;
        int col = idx % feature_w;
        std::vector<float> bbox_pred(
            outs[i + fpn_stride_.size()].begin() + idx * 4 * reg_max,
            outs[i + fpn_stride_.size()].begin() +
                (idx + 1) * 4 * reg_max);
        bbox_results[cur_label].push_back(
            disPred2Bbox(bbox_pred, cur_label, score, col, row,
                               fpn_stride_[i], resize_shape, reg_max));
      }
    }
  }
  for (int i = 0; i < bbox_results.size(); i++) {
    bool flag = bbox_results[i].size() <= 0;
  }
  for (int i = 0; i < bbox_results.size(); i++) {
    bool flag = bbox_results[i].size() <= 0;
    if (bbox_results[i].size() <= 0) {
      continue;
    }
    nms(bbox_results[i], nms_threshold_);
    for (auto box : bbox_results[i]) {
      box.box[0] = box.box[0] / scale_factor_w;
      box.box[2] = box.box[2] / scale_factor_w;
      box.box[1] = box.box[1] / scale_factor_h;
      box.box[3] = box.box[3] / scale_factor_h;
      results.push_back(box);
    }
  }
  return results;
}
std::vector<OCR::StructurePredictResult> LayoutDetector::detect(Mat& srcimg)
{
	int h = srcimg.rows;
	int w = srcimg.cols;
	Mat dstimg = this->preprocess(srcimg);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, dstimg.rows, dstimg.cols };
	
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = net->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	std::cout << "output_names.size(): " << output_names.size() << std::endl;
	
	std::vector<std::vector<float>> out_tensor_list;
	std::vector<std::vector<int>> output_shape_list;
	for(int j = 0; j < output_names.size(); j++){
		const float* floatArray = ort_outputs[j].GetTensorMutableData<float>();
		int outputCount = 1;
        std::cout << ort_outputs.at(j).GetTensorTypeAndShapeInfo().GetShape().size() << std::endl;
        std::vector<int> output_shape;
		for(int i=0; i < ort_outputs.at(j).GetTensorTypeAndShapeInfo().GetShape().size(); i++){
			int dim = ort_outputs.at(j).GetTensorTypeAndShapeInfo().GetShape().at(i);
			outputCount *= dim;
            output_shape.push_back(dim);
		}
        
		Mat binary(dstimg.rows, dstimg.cols, CV_32FC1);
		memcpy(binary.data, floatArray, outputCount * sizeof(float));

		std::vector<float> tensor_data(floatArray, floatArray + outputCount);
		out_tensor_list.push_back(tensor_data);
        output_shape_list.push_back(output_shape);
	}
   
    // post process
    std::vector<int> bbox_num;
    int reg_max = 0;
    for (int i = 0; i < out_tensor_list.size(); i++) {
        if (i == this->fpn_stride_.size()) {
            reg_max = output_shape_list[i][2] / 4;
            break;
        }
    }
    std::vector<int> ori_shape = {srcimg.rows, srcimg.cols};
    std::vector<int> resize_shape = {dstimg.rows, dstimg.cols};
    std::vector<OCR::StructurePredictResult> res;
    res = extract_result(out_tensor_list, ori_shape, resize_shape,
                            reg_max);
    return res;
}


vector< vector<Point2f> > LayoutDetector::order_points_clockwise(vector< vector<Point2f> > results)
{
	vector< vector<Point2f> > order_points(results);
	for (int i = 0; i < results.size(); i++)
	{
		float max_sum_pts = -10000;
		float min_sum_pts = 10000;
		float max_diff_pts = -10000;
		float min_diff_pts = 10000;

		int max_sum_pts_id = 0;
		int min_sum_pts_id = 0;
		int max_diff_pts_id = 0;
		int min_diff_pts_id = 0;
		for (int j = 0; j < 4; j++)
		{
			const float sum_pt = results[i][j].x + results[i][j].y;
			if (sum_pt > max_sum_pts)
			{
				max_sum_pts = sum_pt;
				max_sum_pts_id = j;
			}
			if (sum_pt < min_sum_pts)
			{
				min_sum_pts = sum_pt;
				min_sum_pts_id = j;
			}

			const float diff_pt = results[i][j].y - results[i][j].x;
			if (diff_pt > max_diff_pts)
			{
				max_diff_pts = diff_pt;
				max_diff_pts_id = j;
			}
			if (diff_pt < min_diff_pts)
			{
				min_diff_pts = diff_pt;
				min_diff_pts_id = j;
			}
		}
		order_points[i][0].x = results[i][min_sum_pts_id].x;
		order_points[i][0].y = results[i][min_sum_pts_id].y;
		order_points[i][2].x = results[i][max_sum_pts_id].x;
		order_points[i][2].y = results[i][max_sum_pts_id].y;

		order_points[i][1].x = results[i][min_diff_pts_id].x;
		order_points[i][1].y = results[i][min_diff_pts_id].y;
		order_points[i][3].x = results[i][max_diff_pts_id].x;
		order_points[i][3].y = results[i][max_diff_pts_id].y;
	}
	return order_points;
}

void LayoutDetector::draw_pred(Mat& srcimg, vector< vector<Point2f> > results)
{
	for (int i = 0; i < results.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			circle(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), 2, Scalar(0, 0, 255), -1);
			if (j < 3)
			{
				line(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), Point((int)results[i][j + 1].x, (int)results[i][j + 1].y), Scalar(0, 255, 0));
			}
			else
			{
				line(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), Point((int)results[i][0].x, (int)results[i][0].y), Scalar(0, 255, 0));
			}
		}
	}
}

