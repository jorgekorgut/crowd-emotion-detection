using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class Controller : MonoBehaviour
{
    public RawImage outputImage;
    private Webcam webcam;
    private FaceDetector faceDetector;
    void Start()
    {
        int width = outputImage.texture.width;
        int height =  outputImage.texture.height;

        this.webcam = new Webcam();
        this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8-lite-s.onnx", 0.45f, 0.5f, width, height);

        outputImage.texture = this.webcam.texture;
    }

    void Update()
    {
        processFrame();
    }

    private void processFrame()
    {
        Mat matImage = ImageUtils.ConvertWebCamTextureToMat(webcam.texture, DepthType.Cv8U, 8, 4);
        faceDetector.Detect(matImage);
    }
}