using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Drawing;
using System.Threading.Tasks;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using FaceDetectorFast;

public class Controller : MonoBehaviour
{
    public RawImage outputImage;
    private Webcam webcam;
    private FaceDetector faceDetector;
    private EmotionDetector emotionDetector;
    private Mat detectedFace;
    private int detectionUpdateRateMs = 500;

    private int lastTimeUpdateDetection = 0;

    private List<Face> faces;

    private Mat[] emotionImages;

    void Start()
    {
        this.webcam = new Webcam();
        //this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8-lite-s.onnx", 0.45f, 0.5f);
        //this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8n-face-lindevs.onnx", 0.45f, 0.5f);
        this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov11n-face.onnx", 0.45f, 0.5f);

        this.emotionDetector = new EmotionDetector("Assets/Resources/EmotionDetection/mobilevit_va_mtl.onnx");

        // Load the emotions images
        emotionImages = FileUtils.LoadEmotionsImages();

        //outputImage.texture = new Texture2D(width, height);
        //outputImage.texture = this.webcam.texture;
        //processFrame();
    }
    void Update()
    {
        Mat matImage = ImageUtils.ConvertWebCamTextureToMat(webcam.texture, DepthType.Cv8U, 8, 4);
        // process frame every detectionUpdateRateMs with a the current Time
        if (Time.time * 1000 - lastTimeUpdateDetection > detectionUpdateRateMs)
        {
            faces = faceDetector.Detect(matImage);

            if (faces != null)
            {
                // Parallel for each face detected, detect the emotion
                Parallel.ForEach(faces, face =>
                {
                    int x = (int)face.bbox.lt.x;
                    int y = (int)face.bbox.rb.y; // y starts from the bottom of the image
                    int width = (int)(face.bbox.rb.x - face.bbox.lt.x);
                    int height = (int)(face.bbox.lt.y - face.bbox.rb.y);

                    Rectangle bbox = new Rectangle(x, y, width, height);

                    Mat detectedFace = ImageUtils.CropImage(matImage, bbox);

                    Emotion emotion = emotionDetector.Detect(detectedFace);

                    face.emotion = emotion;
                });

            }

            lastTimeUpdateDetection = (int)(Time.time * 1000);
        }

        drawFaces(matImage);
    }

    private void drawFaces(Mat matImage)
    {
        if (faces == null)
        {
            return;
        }

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
            foreach (Point2D landmark in face.landmark.points)
            {
                CvInvoke.Circle(matImage, new System.Drawing.Point((int)landmark.x, (int)landmark.y), 2, new MCvScalar(0, 0, 255), 2);
            }

            //Draw the emotion over the face
            // Get the icon png from /Assets/Resources/EmotionImages load image


            if (face.emotion != null)
            {
                // draw the emotion icon
                int emotionId = face.emotion.GetEmotion();
                Mat emotionImage = emotionImages[emotionId];

                // Resize emotionImage to have the same width - position as the original image
                CvInvoke.Resize(emotionImage, emotionImage, new Size((int)(width - x), (int)height));

                string emotionImageInfo = "emotionImage: " + emotionImage.Size;
                Debug.Log(emotionImageInfo);
                //CvInvoke.PutText(matImage, face.emotion.GetEmotionText(), new Point((int)x, (int)(y) - 10), FontFace.HersheySimplex, -1, new MCvScalar(0, 0, 255), 2);
            }
        }

        //ImageUtils.ConvertMatToTexture(matImage);

        // Print mat dimensions
        //Debug.Log(testImage.Size);
        outputImage.texture = ImageUtils.ConvertMatToTexture(matImage);

        //outputImage.texture = webcam.texture;
        //outputImage.texture = 
        //Debug.Log(matImage.Size);
        //faceDetector.Detect(matImage);
    }
}