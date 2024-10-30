using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Controller : MonoBehaviour
{
    public RawImage outputImage;
    private Webcam webcam;
    private FaceDetector faceDetector;
    void Start()
    {
        int width = outputImage.texture.width;
        int height =  outputImage.texture.height;

        this.webcam = new Webcam(outputImage);
        this.faceDetector = new FaceDetector("Resources/FaceDetection/yolov8-lite-s.onnx", 0.45f, 0.5f, width, height);
    }

    void Update()
    {
        //this.webcam.Update();
    }
}