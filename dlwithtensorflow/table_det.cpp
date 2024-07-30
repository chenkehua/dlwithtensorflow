#include"table_det.h"

float calculate_iou2(std::vector<float> &box1, std::vector<float> &box2) {
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

void nms2(std::vector<OCR::StructurePredictResult> &input_boxes,
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
      float iou = calculate_iou2(input_boxes[i].box, input_boxes[j].box);
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

float fast_exp2(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}
std::vector<float> activation_function_softmax2(std::vector<float> &src) {
  int length = src.size();
  std::vector<float> dst;
  dst.resize(length);
  const float alpha = float(*std::max_element(&src[0], &src[0 + length]));
  float denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp2(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return dst;
}

TableDetector::TableDetector()
{
	this->binaryThreshold = 0.3;
	this->polygonThreshold = 0.5;
	this->unclipRatio = 1.6;
	this->maxCandidates = 1000;
	string model_path = "inference/ch_ppstructure_mobile_v2.0_SLANet_infer/model.onnx";
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

Mat TableDetector::preprocess(Mat srcimg)
{
    int max_len = 488;
	Mat dstimg;
	// cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	// resize(dstimg, dstimg, Size(608, 800));
    int w = srcimg.cols;
    int h = srcimg.rows;
    int max_wh = w >= h ? w : h;
    float ratio = w >= h ? float(max_len) / float(w) : float(max_len) / float(h);
    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);
    //cv::resize(dstimg, dstimg, cv::Size(resize_w, resize_h));
    cv::resize(srcimg, dstimg, cv::Size(resize_w, resize_h));

	return dstimg;
}

//void TableDetector::normalize_(Mat img)
void TableDetector::normalize_(Mat *im)
{
	//    img.convertTo(img, CV_32F);
  /*
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
  */
  double e = 1.0;
  bool is_scale =true;
  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(*im, bgr_channels);
  for (auto i = 0; i < bgr_channels.size(); i++) {
    bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale_[i],
                              (0.0 - mean_[i]) * scale_[i]);
  }
  cv::merge(bgr_channels, *im);
  /*
  int row = im->rows;
  int col = im->cols;
  this->input_image_.resize(row * col * im->channels());
  for (int c = 0; c < 3; c++)
  {
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = im->ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
  }*/
}

OCR::StructurePredictResult disPred2Bbox2(std::vector<float> bbox_pred, int label,
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
    std::vector<float> dis_after_sm = activation_function_softmax2(bbox_pred_i);
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

std::vector<std::string> ReadDict(const std::string &path) {
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

void extract_table(
    std::vector<float> &loc_preds, std::vector<float> &structure_probs,
    std::vector<float> &rec_scores, std::vector<int> &loc_preds_shape,
    std::vector<int> &structure_probs_shape,
    std::vector<std::vector<std::string>> &rec_html_tag_batch,
    std::vector<std::vector<std::vector<int>>> &rec_boxes_batch,
    std::vector<int> &width_list, std::vector<int> &height_list){
  std::vector<std::string> label_list_ = ReadDict("table_structure_dict_ch.txt");
  std::string end = "eos";
  std::string beg = "sos";
  if (1) {
    label_list_.push_back("<td></td>");
    std::vector<std::string>::iterator it;
    for (it = label_list_.begin(); it != label_list_.end();) {
      if (*it == "<td>") {
        it = label_list_.erase(it);
      } else {
        ++it;
      }
    }
  }
  label_list_.insert(label_list_.begin(), beg);
  label_list_.push_back(end);

  for (int batch_idx = 0; batch_idx < structure_probs_shape[0]; batch_idx++) {
    // image tags and boxs
    std::vector<std::string> rec_html_tags;
    std::vector<std::vector<int>> rec_boxes;

    float score = 0.f;
    int count = 0;
    float char_score = 0.f;
    int char_idx = 0;

    // step
    for (int step_idx = 0; step_idx < structure_probs_shape[1]; step_idx++) {
      std::string html_tag;
      std::vector<int> rec_box;
      // html tag
      int step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
                           structure_probs_shape[2];
      char_idx = int(argmax(
          &structure_probs[step_start_idx],
          &structure_probs[step_start_idx + structure_probs_shape[2]]));
      char_score = float(*std::max_element(
          &structure_probs[step_start_idx],
          &structure_probs[step_start_idx + structure_probs_shape[2]]));
      html_tag = label_list_[char_idx];

      if (step_idx > 0 && html_tag == end) {
        break;
      }
      if (html_tag == beg) {
        continue;
      }
      count += 1;
      score += char_score;
      rec_html_tags.push_back(html_tag);

      // box
      if (html_tag == "<td>" || html_tag == "<td" || html_tag == "<td></td>") {
        for (int point_idx = 0; point_idx < loc_preds_shape[2]; point_idx++) {
          step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
                               loc_preds_shape[2] +
                           point_idx;
          float point = loc_preds[step_start_idx];
          if (point_idx % 2 == 0) {
            point = int(point * width_list[batch_idx]);
          } else {
            point = int(point * height_list[batch_idx]);
          }
          rec_box.push_back(point);
        }
        rec_boxes.push_back(rec_box);
      }
    }
    score /= count;
    if (std::isnan(score) || rec_boxes.size() == 0) {
      score = -1;
    }
    rec_scores.push_back(score);
    rec_boxes_batch.push_back(rec_boxes);
    rec_html_tag_batch.push_back(rec_html_tags);
  }
}

void TableDetector::padding(const cv::Mat &img, cv::Mat &resize_img,
                      const int max_len) {
  int w = img.cols;
  int h = img.rows;
  cv::copyMakeBorder(img, resize_img, 0, max_len - h, 0, max_len - w,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void Permute2(const cv::Mat *im, float *data) {
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

void TableDetector::detect(Mat& srcimg,std::vector<std::vector<std::string>> &structure_html_tags,std::vector<float> &structure_scores,std::vector<std::vector<std::vector<int>>> &structure_boxes)
{
    std::vector<int> width_list;
    std::vector<int> height_list;

	int h = srcimg.rows;
	int w = srcimg.cols;
    width_list.push_back(srcimg.cols);
    height_list.push_back(srcimg.rows);
    // resize
	Mat dstimg = this->preprocess(srcimg);
    cv::Mat pad_img;
    int max_len = 488;
    this->padding(dstimg, pad_img, max_len);

	//this->normalize_(pad_img);

	this->normalize_(&pad_img);
	//array<int64_t, 4> input_shape_{ 1, 3, dstimg.rows, dstimg.cols };
	//array<int64_t, 4> input_shape_{ 1, 3, pad_img.rows, pad_img.cols };
	array<int64_t, 4> input_shape_{ 1, 3, max_len, max_len };

    std::vector<float> input(1 * 3 * max_len * max_len, 0.0f);
    Permute2(&pad_img, input.data());
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	//Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input.data(), input.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = net->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	std::vector<std::vector<float>> out_tensor_list;
	std::vector<std::vector<int>> output_shape_list;
	for(int j = 0; j < output_names.size(); j++){
		const float* floatArray = ort_outputs[j].GetTensorMutableData<float>();
		int outputCount = 1;
        std::vector<int> output_shape;
		for(int i=0; i < ort_outputs.at(j).GetTensorTypeAndShapeInfo().GetShape().size(); i++){
			int dim = ort_outputs.at(j).GetTensorTypeAndShapeInfo().GetShape().at(i);
			outputCount *= dim;
            output_shape.push_back(dim);
            std::cout << "dim: " << dim << std::endl;
		}

		std::vector<float> loc_preds(floatArray, floatArray + outputCount);
		out_tensor_list.push_back(loc_preds);
        output_shape_list.push_back(output_shape);
	}
    std::vector<float> loc_preds = out_tensor_list[0];
    std::vector<float> structure_probs = out_tensor_list[1];
    std::vector<int> predict_shape0 = output_shape_list[0];
    std::vector<int> predict_shape1 = output_shape_list[1];
    
    //result
    std::vector<float> structure_score_batch;
    std::vector<std::vector<std::vector<int>>> structure_boxes_batch;
    std::vector<std::vector<std::string>> structure_html_tag_batch;

    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << loc_preds.size() << std::endl;
    std::cout << structure_probs.size() << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;
    
    // extract table result
    extract_table(loc_preds, structure_probs, structure_score_batch,
                              predict_shape0, predict_shape1,
                              structure_html_tag_batch, structure_boxes_batch,
                              width_list, height_list);

    for (int m = 0; m < predict_shape0[0]; m++) {

      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(), "<table>");
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(), "<body>");
      structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(), "<html>");
      structure_html_tag_batch[m].push_back("</table>");
      structure_html_tag_batch[m].push_back("</body>");
      structure_html_tag_batch[m].push_back("</html>");
      structure_html_tags.push_back(structure_html_tag_batch[m]);
      structure_scores.push_back(structure_score_batch[m]);
      structure_boxes.push_back(structure_boxes_batch[m]);
    }
    /*
    for(int i =0; i < structure_html_tags.size(); i++){
        for(int j =0; j < structure_html_tags[i].size(); j++){
            std::cout << structure_html_tags[i][j] << std::endl;
        }
    }*/
}


vector< vector<Point2f> > TableDetector::order_points_clockwise(vector< vector<Point2f> > results)
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

void TableDetector::draw_pred(Mat& srcimg, vector< vector<Point2f> > results)
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

float TableDetector::contourScore(const Mat& binary, const vector<Point>& contour)
{
	Rect rect = boundingRect(contour);
	int xmin = max(rect.x, 0);
	int xmax = min(rect.x + rect.width, binary.cols - 1);
	int ymin = max(rect.y, 0);
	int ymax = min(rect.y + rect.height, binary.rows - 1);

	Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

	Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
	vector<Point> roiContour;
	for (size_t i = 0; i < contour.size(); i++) {
		Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
		roiContour.push_back(pt);
	}
	vector<vector<Point>> roiContours = { roiContour };
	fillPoly(mask, roiContours, Scalar(1));
	float score = mean(binROI, mask).val[0];
	return score;
}

void TableDetector::unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly)
{
	float area = contourArea(inPoly);
	float length = arcLength(inPoly, true);
	float distance = area * unclipRatio / length;

	size_t numPoints = inPoly.size();
	vector<vector<Point2f>> newLines;
	for (size_t i = 0; i < numPoints; i++)
	{
		vector<Point2f> newLine;
		Point pt1 = inPoly[i];
		Point pt2 = inPoly[(i - 1) % numPoints];
		Point vec = pt1 - pt2;
		float unclipDis = (float)(distance / norm(vec));
		Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
		newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
		newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
		newLines.push_back(newLine);
	}

	size_t numLines = newLines.size();
	for (size_t i = 0; i < numLines; i++)
	{
		Point2f a = newLines[i][0];
		Point2f b = newLines[i][1];
		Point2f c = newLines[(i + 1) % numLines][0];
		Point2f d = newLines[(i + 1) % numLines][1];
		Point2f pt;
		Point2f v1 = b - a;
		Point2f v2 = d - c;
		float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

		if (fabs(cosAngle) > 0.7)
		{
			pt.x = (b.x + c.x) * 0.5;
			pt.y = (b.y + c.y) * 0.5;
		}
		else
		{
			float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
				d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
			float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
			float s = num / denom;

			pt.x = a.x + s * (b.x - a.x);
			pt.y = a.y + s * (b.y - a.y);
		}
		outPoly.push_back(pt);
	}
}

Mat TableDetector::get_rotate_crop_image(const Mat& frame, vector<Point2f> vertices)
{
	Rect rect = boundingRect(Mat(vertices));
	Mat crop_img = frame(rect);

	const Size outputSize = Size(rect.width, rect.height);

	vector<Point2f> targetVertices{ Point2f(0, outputSize.height),Point2f(0, 0), Point2f(outputSize.width, 0), Point2f(outputSize.width, outputSize.height)};

	for (int i = 0; i < 4; i++)
	{
		vertices[i].x -= rect.x;
		vertices[i].y -= rect.y;
	}
	
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(crop_img, result, rotationMatrix, outputSize, cv::BORDER_REPLICATE);
	return result;
}

std::vector<int> xyxyxyxy2xyxy(std::vector<std::vector<int>> &box) {
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  std::vector<int> box1(4, 0);
  box1[0] = left;
  box1[1] = top;
  box1[2] = right;
  box1[3] = bottom;
  return box1;
}
std::vector<int> xyxyxyxy2xyxy(std::vector<int> &box) {
  int x_collect[4] = {box[0], box[2], box[4], box[6]};
  int y_collect[4] = {box[1], box[3], box[5], box[7]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  std::vector<int> box1(4, 0);
  box1[0] = left;
  box1[1] = top;
  box1[2] = right;
  box1[3] = bottom;
  return box1;
}

float iou(std::vector<int> &box1, std::vector<int> &box2) {
  int area1 = std::max(0, box1[2] - box1[0]) * std::max(0, box1[3] - box1[1]);
  int area2 = std::max(0, box2[2] - box2[0]) * std::max(0, box2[3] - box2[1]);

  // computing the sum_area
  int sum_area = area1 + area2;

  // find the each point of intersect rectangle
  int x1 = std::max(box1[0], box2[0]);
  int y1 = std::max(box1[1], box2[1]);
  int x2 = std::min(box1[2], box2[2]);
  int y2 = std::min(box1[3], box2[3]);

  // judge if there is an intersect
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    int intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

float dis(std::vector<int> &box1, std::vector<int> &box2) {
  int x1_1 = box1[0];
  int y1_1 = box1[1];
  int x2_1 = box1[2];
  int y2_1 = box1[3];

  int x1_2 = box2[0];
  int y1_2 = box2[1];
  int x2_2 = box2[2];
  int y2_2 = box2[3];

  float dis =
      abs(x1_2 - x1_1) + abs(y1_2 - y1_1) + abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  float dis_2 = abs(x1_2 - x1_1) + abs(y1_2 - y1_1);
  float dis_3 = abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  return dis + std::min(dis_2, dis_3);
}

static bool comparison_dis(const std::vector<float> &dis1,
                         const std::vector<float> &dis2) {
    if (dis1[1] < dis2[1]) {
        return true;
    } else if (dis1[1] == dis2[1]) {
        return dis1[0] < dis2[0];
    } else {
        return false;
    }
}

std::string TableDetector::rebuild_table(std::vector<std::string> structure_html_tags,
                               std::vector<std::vector<int>> structure_boxes,
                               std::vector<OCR::OCRPredictResult> &ocr_result) {
  // match text in same cell
  std::vector<std::vector<std::string>> matched(structure_boxes.size(),
                                                std::vector<std::string>());
  std::vector<int> ocr_box;
  std::vector<int> structure_box;
  for (int i = 0; i < ocr_result.size(); i++) {
    ocr_box = xyxyxyxy2xyxy(ocr_result[i].box);
    ocr_box[0] -= 1;
    ocr_box[1] -= 1;
    ocr_box[2] += 1;
    ocr_box[3] += 1;
    std::vector<std::vector<float>> dis_list(structure_boxes.size(),
                                             std::vector<float>(3, 100000.0));
    for (int j = 0; j < structure_boxes.size(); j++) {
      if (structure_boxes[j].size() == 8) {
        structure_box = xyxyxyxy2xyxy(structure_boxes[j]);
      } else {
        structure_box = structure_boxes[j];
      }
      dis_list[j][0] = dis(ocr_box, structure_box);
      dis_list[j][1] = 1 - iou(ocr_box, structure_box);
      dis_list[j][2] = j;
    }
    // find min dis idx
    std::sort(dis_list.begin(), dis_list.end(),
              comparison_dis);
    matched[dis_list[0][2]].push_back(ocr_result[i].text);
  }

  // get pred html
  std::string html_str = "";
  int td_tag_idx = 0;
  for (int i = 0; i < structure_html_tags.size(); i++) {
    if (structure_html_tags[i].find("</td>") != std::string::npos) {
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "<td>";
      }
      if (matched[td_tag_idx].size() > 0) {
        bool b_with = false;
        if (matched[td_tag_idx][0].find("<b>") != std::string::npos &&
            matched[td_tag_idx].size() > 1) {
          b_with = true;
          html_str += "<b>";
        }
        for (int j = 0; j < matched[td_tag_idx].size(); j++) {
          std::string content = matched[td_tag_idx][j];
          if (matched[td_tag_idx].size() > 1) {
            // remove blank, <b> and </b>
            if (content.length() > 0 && content.at(0) == ' ') {
              content = content.substr(0);
            }
            if (content.length() > 2 && content.substr(0, 3) == "<b>") {
              content = content.substr(3);
            }
            if (content.length() > 4 &&
                content.substr(content.length() - 4) == "</b>") {
              content = content.substr(0, content.length() - 4);
            }
            if (content.empty()) {
              continue;
            }
            // add blank
            if (j != matched[td_tag_idx].size() - 1 &&
                content.at(content.length() - 1) != ' ') {
              content += ' ';
            }
          }
          html_str += content;
        }
        if (b_with) {
          html_str += "</b>";
        }
      }
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "</td>";
      } else {
        html_str += structure_html_tags[i];
      }
      td_tag_idx += 1;
    } else {
      html_str += structure_html_tags[i];
    }
  }
  return html_str;
}
