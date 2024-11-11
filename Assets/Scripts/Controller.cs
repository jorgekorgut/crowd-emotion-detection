using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Drawing;

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
        this.webcam = new Webcam();
        this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8-lite-s.onnx", 0.45f, 0.5f);

        //outputImage.texture = new Texture2D(width, height);
        //outputImage.texture = this.webcam.texture;
        //processFrame();
    }

    private int counter = 0;
    void Update()
    {
        if (counter == 100)
        {
            processFrame();
        }
        counter++;
        //processFrame();
    }

    private void processFrame()
    {

        Mat matImage = ImageUtils.ConvertWebCamTextureToMat(webcam.texture, DepthType.Cv8U, 8, 4);
        List<Face> faces = faceDetector.Detect(matImage);

        foreach (Face face in faces)
        {
            float x = face.bbox.lt.x;
            float y = face.bbox.lt.y;

            float width = face.bbox.rb.x - face.bbox.lt.x;
            float height = face.bbox.rb.y - face.bbox.lt.y;

            //Construct a rectangle from the bounding box of face
            Rectangle rect = new Rectangle((int)x, (int)y, (int)width, (int)height);

            CvInvoke.Rectangle(matImage, rect, new MCvScalar(0, 255, 0), 2);
        }
        outputImage.texture = ImageUtils.ConvertMatToTexture(matImage);

        //outputImage.texture = webcam.texture;
        //outputImage.texture = 
        //Debug.Log(matImage.Size);
        //faceDetector.Detect(matImage);
    }
}