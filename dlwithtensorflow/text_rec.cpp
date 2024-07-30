#include"text_rec.h"

TextRecognizer::TextRecognizer()
{
	string model_path = "inference/ch_PP-OCRv4_rec_server_infer/model.onnx";
	//std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

	ifstream ifs("rec_word_dict.txt");
	string line;
    this->alphabet.push_back("#");
	while (getline(ifs, line))
	{
		this->alphabet.push_back(line);
	}
	this->alphabet.push_back(" ");
	names_len = this->alphabet.size();
}

Mat TextRecognizer::preprocess(Mat srcimg)
{
	Mat dstimg;
	int h = srcimg.rows;
	int w = srcimg.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inpHeight * ratio));
	if (ceil(this->inpHeight*ratio) > this->inpWidth)
	{
		resized_w = this->inpWidth;
	}
	
	resize(srcimg, dstimg, Size(resized_w, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

void TextRecognizer::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(this->inpHeight * this->inpWidth * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < inpWidth; j++)
			{
				if (j < col)
				{
					float pix = img.ptr<uchar>(i)[j * 3 + c];
					this->input_image_[c * row * inpWidth + i * inpWidth + j] = (pix / 255.0 - 0.5) / 0.5;
				}
				else
				{
					this->input_image_[c * row * inpWidth + i * inpWidth + j] = 0;
				}
			}
		}
	}
}

string TextRecognizer::predict_text(Mat cv_image)
{
	Mat dstimg = this->preprocess(cv_image);
	this->normalize_(dstimg);
	
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	
	int i = 0, j = 0;
	int h = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
	int w = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
	
	preb_label.resize(w);
	for (i = 0; i < w; i++)
	{
		int one_label_idx = 0;
		float max_data = -10000;
		for (j = 0; j < h; j++)
		{
			float data_ = pdata[i*h + j];
			if (data_ > max_data)
			{
				max_data = data_;
				one_label_idx = j;
			}
		}
		preb_label[i] = one_label_idx;
	}
	
	vector<int> no_repeat_blank_label;
	for (size_t elementIndex = 0; elementIndex < w; ++elementIndex)
	{
		if (preb_label[elementIndex] != 0 && !(elementIndex > 0 && preb_label[elementIndex - 1] == preb_label[elementIndex]))
		{
			no_repeat_blank_label.push_back(preb_label[elementIndex] - 1);
		}
	}
	
	int len_s = no_repeat_blank_label.size();
	string plate_text;
	for (i = 0; i < len_s; i++)
	{
		plate_text += alphabet[no_repeat_blank_label[i]];
	}
	
	return plate_text;
}

std::vector<int> argsort(const std::vector<float> &array) {
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i)
    array_index[i] = i;

  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

void Normalize(cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &scale, const bool is_scale) {
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(*im, bgr_channels);
  for (auto i = 0; i < bgr_channels.size(); i++) {
    bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale[i],
                              (0.0 - mean[i]) * scale[i]);
  }
  cv::merge(bgr_channels, *im);
}

void PermuteBatch(const std::vector<cv::Mat> imgs, float *data) {
  for (int j = 0; j < imgs.size(); j++) {
    int rh = imgs[j].rows;
    int rw = imgs[j].cols;
    int rc = imgs[j].channels();
    for (int i = 0; i < rc; ++i) {
      cv::extractChannel(
          imgs[j], cv::Mat(rh, rw, CV_32FC1, data + (j * rc + i) * rh * rw), i);
    }
  }
}
void CrnnResizeImg(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                        bool use_tensorrt,
                        const std::vector<int> &rec_image_shape) {
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  imgW = int(imgH * wh_ratio);

  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                     int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                     {0, 0, 0});
}

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

void TextRecognizer::batch_predict_text(std::vector<cv::Mat> img_list,std::vector<std::string> &rec_texts,std::vector<float> &rec_text_scores){
    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++) {
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<int> indices = argsort(width_list);
    int rec_batch_num_ = 12;
    int rec_img_h_ = 48;
    int rec_img_w_ = 320;
    std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};

    for (int beg_img_no = 0; beg_img_no < img_num;beg_img_no += rec_batch_num_) {
        int end_img_no = std::min(img_num, beg_img_no + rec_batch_num_);   
        int batch_num = end_img_no - beg_img_no;
        int imgH = rec_image_shape_[1];
        int imgW = rec_image_shape_[2];
        float max_wh_ratio = imgW * 1.0 / imgH;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            int h = img_list[indices[ino]].rows;
            int w = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }
        int batch_width = imgW;
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
            std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
            bool is_scale_ = true;
            bool use_tensorrt_ = false;
            CrnnResizeImg(srcimg, resize_img, max_wh_ratio,use_tensorrt_, rec_image_shape_);
            Normalize(&resize_img, mean_, scale_,is_scale_);
            norm_img_batch.push_back(resize_img);
            batch_width = std::max(resize_img.cols, batch_width);
        }
        std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
        PermuteBatch(norm_img_batch, input.data());
	    array<int64_t, 4> input_shape_{ batch_num, 3, 48, 320 };

	    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input.data(), input.size(), input_shape_.data(), input_shape_.size());
	    vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
        
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
		    }
		    std::vector<float> loc_preds(floatArray, floatArray + outputCount);
		    out_tensor_list.push_back(loc_preds);
            output_shape_list.push_back(output_shape);
        }
        int predict_shape0 = output_shape_list[0][0];
        int predict_shape1 = output_shape_list[0][1];
        int predict_shape2 = output_shape_list[0][2];
        std::vector<float> pdata_d = out_tensor_list[0];
        float* pdata = pdata_d.data();
        for (int m = 0; m < predict_shape0; m++) {
          std::string str_res;
          int argmax_idx;
          int last_index = 0;
          float score = 0.f;
          int count = 0;
          float max_value = 0.0f;

          for (int n = 0; n < predict_shape1; n++) {
            // get idx
            argmax_idx = int(argmax(
                &pdata[(m * predict_shape1 + n) * predict_shape2],
                &pdata[(m * predict_shape1 + n + 1) * predict_shape2]));
            // get score
            max_value = float(*std::max_element(
                &pdata[(m * predict_shape1 + n) * predict_shape2],
                &pdata[(m * predict_shape1 + n + 1) * predict_shape2]));

            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
              score += max_value;
              count += 1;
              str_res += alphabet[argmax_idx];
            }
            last_index = argmax_idx;
          }
          score /= count;
          if (std::isnan(score)) {
            continue;
          }
          rec_texts[indices[beg_img_no + m]] = str_res;
          rec_text_scores[indices[beg_img_no + m]] = score;
        }
    }
    for(auto item : rec_texts){
        std::cout << "result: " << item << std::endl;
    }

}
