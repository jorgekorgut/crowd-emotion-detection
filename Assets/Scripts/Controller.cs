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

    private Mat testImage;

    void Start()
    {
        this.webcam = new Webcam();
        this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8-lite-s.onnx", 0.45f, 0.5f);

        // Load jpg image on unity and convert it to mat
        testImage = ImageUtils.LoadJPGToMat("Assets/Resources/Images/femme.jpg");

        //outputImage.texture = new Texture2D(width, height);
        //outputImage.texture = this.webcam.texture;
        //processFrame();
    }

    private int counter = 0;
    void Update()
    {
        /*
        if (counter == 100)
        {
            processFrame();
        }
        counter++;
        */
        processFrame();
    }

    private void processFrame()
    {
        Mat matImage = ImageUtils.ConvertWebCamTextureToMat(webcam.texture, DepthType.Cv8U, 8, 4);
        //Mat matImage = testImage;
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

            //Print the landmakrs of the face
            foreach (Point landmark in face.landmark.points)
            {
                CvInvoke.Circle(matImage, new System.Drawing.Point((int)landmark.x, (int)landmark.y), 2, new MCvScalar(0, 0, 255), 2);
            }
        }
        
        // Print mat dimensions
        //Debug.Log(testImage.Size);
        outputImage.texture = ImageUtils.ConvertMatToTexture(matImage);

        //outputImage.texture = webcam.texture;
        //outputImage.texture = 
        //Debug.Log(matImage.Size);
        //faceDetector.Detect(matImage);
    }
}