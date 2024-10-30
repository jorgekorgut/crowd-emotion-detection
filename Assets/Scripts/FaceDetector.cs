using UnityEngine;
using System;

using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Util;

using System.Drawing;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

class FaceDetector
{
    private int inputNumberOfChannels = 3;
    private int inputWidth;
    private int inputHeight;

    private float confThreshold;
    private float nmsThreshold;
    private bool isLoaded = false;
    private Net net;
    public FaceDetector(string modelpath, float confThreshold, float nmsThreshold, int width, int height)
    {
        this.confThreshold = confThreshold;
        this.nmsThreshold = nmsThreshold;
        this.inputWidth = width;
        this.inputHeight = height;

        readNet(modelpath);
    }

    private void readNet(string path)
    {
        try
        {
            this.net = DnnInvoke.ReadNetFromONNX(path);
            isLoaded = true;
        }
        catch (Exception e)
        {
            Debug.Log(e.Message);
        }
    }

    //public void detect(Mat& frame)
    public void Detect(Mat frame)
    {
        string[] outputLayers = new string[]
        {
            "scores", "boxes"
        };

        if (!isLoaded)
        {
            Debug.Log("Model is not loaded.");
            return;
        }

        Mat preprocessedFrame = Preprocess(frame);

        Debug.Log(preprocessedFrame.Size);

        VectorOfMat netOutput = new VectorOfMat();
        net.SetInput(preprocessedFrame);
        //net.Forward(netOutput, outputLayers);

        Postprocess(netOutput);

        // int newh = 0, neww = 0, padh = 0, padw = 0;
        // Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
        // Mat blob;
        // blobFromImage(dst, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
        // this->net.setInput(blob);
        // vector<Mat> outs;
        // ///net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行   
        // this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

        // /////generate proposals
        // vector<Rect> boxes;
        // vector<float> confidences;
        // vector<vector<Point>> landmarks;
        // float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

        // generate_proposal(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
        // generate_proposal(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
        // generate_proposal(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

        // // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // // lower confidences
        // vector<int> indices;
        // NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
        // for (size_t i = 0; i < indices.size(); ++i)
        // {
        //     int idx = indices[i];
        //     Rect box = boxes[idx];
        //     this->drawPred(confidences[idx], box.x, box.y,
        //         box.x + box.width, box.y + box.height, srcimg, landmarks[idx]);
        // }
    }

    private Mat Preprocess(Mat img)
    {
        Mat rgbImage = new Mat(new Size(inputWidth, inputHeight), img.Depth, inputNumberOfChannels);

        var conversion = img.NumberOfChannels == 4 ? ColorConversion.Bgra2Rgb : ColorConversion.Bgr2Rgb;
        CvInvoke.CvtColor(img, rgbImage, conversion);

        Mat inputBlob = DnnInvoke.BlobFromImage(
            rgbImage, 1.0 / 128,
            new Size(inputWidth, inputHeight),
            new MCvScalar(127, 127, 127), true
        );
        
        return inputBlob;
    }

    private void Postprocess(VectorOfMat outBlobs)
    {
        //Debug.Log(outBlobs.Size);
        //Mat confidencesMat = outBlobs[0];
        //Mat boxesMat = outBlobs[1];

        //Debug.Log(boxesMat.ToString());

    }

    //private Mat resize_image(Mat srcimg, int* newh, int* neww, int* padh, int* padw)
    private void resize_image()
    {
        // int srch = srcimg.rows, srcw = srcimg.cols;
        // *newh = this->inpHeight;
        // *neww = this->inpWidth;
        // Mat dstimg;
        // if (this->keep_ratio && srch != srcw)
        // {
        //     float hw_scale = (float)srch / srcw;
        //     if (hw_scale > 1)
        //     {
        //         *newh = this->inpHeight;
        //         *neww = int(this->inpWidth / hw_scale);
        //         resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
        //         *padw = int((this->inpWidth - *neww) * 0.5);
        //         copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, BORDER_CONSTANT, 0);
        //     }
        //     else
        //     {
        //         *newh = (int)this->inpHeight * hw_scale;
        //         *neww = this->inpWidth;
        //         resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
        //         *padh = (int)(this->inpHeight - *newh) * 0.5;
        //         copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, BORDER_CONSTANT, 0);
        //     }
        // }
        // else
        // {
        //     resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
        // }
        // return dstimg;
    }

    private const bool keep_ratio = true;
    private const int num_class = 1;
    private const int reg_max = 16;
    //private Net net;
    //private void softmax_(const float* x, float* y, int length)
    private void softmax()
    {
        //         float sum = 0;
        //     int i = 0;
        //             for (i = 0; i<length; i++)
        //             {
        //                 y[i] = exp(x[i]);
        //     sum += y[i];
        //             }
        // for (i = 0; i < length; i++)
        // {
        //     y[i] /= sum;
        // }
    }

    //private void generate_proposal(Mat out, vector<Rect>& boxes, vector<float>& confidences, vector<vector<Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw)
    private void generate_proposal()
    {
        // const int feat_h = out.size[2];
        // const int feat_w = out.size[3];
        // cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << endl;
        // const int stride = (int)ceil((float)inpHeight / feat_h);
        // const int area = feat_h * feat_w;
        // float* ptr = (float*)out.data;
        // float* ptr_cls = ptr + area * reg_max * 4;
        // float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

        // for (int i = 0; i < feat_h; i++)
        // {
        // 	for (int j = 0; j < feat_w; j++)
        // 	{
        // 		const int index = i * feat_w + j;
        // 		int cls_id = -1;
        // 		float max_conf = -10000;
        // 		for (int k = 0; k < num_class; k++)
        // 		{
        // 			float conf = ptr_cls[k*area + index];
        // 			if (conf > max_conf)
        // 			{
        // 				max_conf = conf;
        // 				cls_id = k;
        // 			}
        // 		}
        // 		float box_prob = sigmoid_x(max_conf);
        // 		if (box_prob > this->confThreshold)
        // 		{
        // 			float pred_ltrb[4];
        // 			float* dfl_value = new float[reg_max];
        // 			float* dfl_softmax = new float[reg_max];
        // 			for (int k = 0; k < 4; k++)
        // 			{
        // 				for (int n = 0; n < reg_max; n++)
        // 				{
        // 					dfl_value[n] = ptr[(k*reg_max + n)*area + index];
        // 				}
        // 				softmax_(dfl_value, dfl_softmax, reg_max);

        // 				float dis = 0.f;
        // 				for (int n = 0; n < reg_max; n++)
        // 				{
        // 					dis += n * dfl_softmax[n];
        // 				}

        // 				pred_ltrb[k] = dis * stride;
        // 			}
        // 			float cx = (j + 0.5f)*stride;
        // 			float cy = (i + 0.5f)*stride;
        // 			float xmin = max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);  ///还原回到原图
        // 			float ymin = max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
        // 			float xmax = min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
        // 			float ymax = min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));
        // 			Rect box = Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
        // 			boxes.push_back(box);
        // 			confidences.push_back(box_prob);

        // 			vector<Point> kpts(5);
        // 			for (int k = 0; k < 5; k++)
        // 			{
        // 				float x = ((ptr_kp[(k * 3)*area + index] * 2 + j)*stride - padw)*ratiow;  ///还原回到原图
        // 				float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i)*stride - padh)*ratioh;
        // 				///float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
        // 				kpts[k] = Point(int(x), int(y));
        // 			}
        // 			landmarks.push_back(kpts);
        // 		}
        // 	}
        // }
    }

    //private void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<Point> landmark);
}