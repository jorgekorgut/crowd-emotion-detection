using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Drawing;
using TMPro;
using System.Threading.Tasks;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using FaceDetectorFast;

public class Controller : MonoBehaviour
{
    public RawImage outputImage;

    public Canvas UI;
    private Webcam webcam;
    private FaceDetector faceDetector;
    private EmotionDetector emotionDetector;
    private Mat detectedFace;
    private int detectionUpdateRateMs = 1000;

    private int lastTimeUpdateDetection = 0;

    private List<Face> faces;

    private Sprite[] emotionImages;

    private float matToTextureCoordinatesX = 1;
    private float matToTextureCoordinatesY = 1;

    void Start()
    {
        this.webcam = new Webcam();
        if(this.webcam.isLoaded)
        {
            outputImage.texture = this.webcam.texture;
        }
        else
        {
            Debug.Log("Webcam not loaded, Loading image from file instead.");
            //string filename = "Assets/Resources/Images/femme.jpg";
            string filename = "Assets/Resources/Images/Student_in_Class_Tulane_University_September_2002.jpg"; // author: https://www.flickr.com/people/28035080@N04
            var rawData = System.IO.File.ReadAllBytes(filename);
            Texture2D texture = new Texture2D(2, 2);
            texture.LoadImage(rawData);
            outputImage.texture = texture;
        }

        //this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8-lite-s.onnx", 0.45f, 0.5f);
        //this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov8n-face-lindevs.onnx", 0.45f, 0.5f);
        this.faceDetector = new FaceDetector("Assets/Resources/FaceDetection/yolov11n-face.onnx", 0.45f, 0.5f);

        this.emotionDetector = new EmotionDetector("Assets/Resources/EmotionDetection/mobilevit_va_mtl.onnx");

        // Load the emotions images
        emotionImages = FileUtils.LoadEmotionsImages();

        matToTextureCoordinatesX = (float)outputImage.rectTransform.rect.width / (float)outputImage.texture.width;
        matToTextureCoordinatesY = (float)outputImage.rectTransform.rect.height / (float)outputImage.texture.height;
        //processFrame();

    }

    void Update()
    {
        // process frame every detectionUpdateRateMs with a the current Time
        if (Time.time * 1000 - lastTimeUpdateDetection > detectionUpdateRateMs)
        {
            Mat matImage = null;
            if(webcam.isLoaded){
                matImage = ImageUtils.ConvertTextureToMat(webcam.texture, DepthType.Cv8U, 8, 4);
            }
            else
            {
                matImage = ImageUtils.ConvertTextureToMat(outputImage.texture as Texture2D, DepthType.Cv8U, 8, 4);
            }

            Task.Run(() =>
            {
                faces = faceDetector.Detect(matImage);

                if (faces != null)
                {
                    // Parallel for each face detected, detect the emotion
                    //Parallel.ForEach(faces, face =>
                    foreach (Face face in faces)
                    {
                        int x = (int)face.bbox.lt.x;
                        int y = (int)face.bbox.rb.y; // y starts from the bottom of the image
                        int width = (int)(face.bbox.rb.x - face.bbox.lt.x);
                        int height = (int)(face.bbox.lt.y - face.bbox.rb.y);

                        Rectangle bbox = new Rectangle(x, y, width, height);

                        Mat detectedFace = ImageUtils.CropImage(matImage, bbox);

                        Emotion emotion = emotionDetector.Detect(detectedFace);

                        face.emotion = emotion;
                    }
                    //);
                }
            });

            drawFaces();
            lastTimeUpdateDetection = (int)(Time.time * 1000);
        }

        //outputImage.texture = ImageUtils.ConvertMatToTexture(matImage);
    }

    private void drawFaces()
    {

        if (faces == null)
        {
            return;
        }

        // Remove all game objects with face tag
        GameObject[] facesGameObject = GameObject.FindGameObjectsWithTag("Face");
        foreach (GameObject face in facesGameObject)
        {
            Destroy(face);
        }

        foreach (Face face in faces)
        {
            //Center of the face
            float x = (face.bbox.rb.x + face.bbox.lt.x)/2;
            float y = (face.bbox.lt.y + face.bbox.rb.y)/2;

            float width =  (face.bbox.rb.x - face.bbox.lt.x);
            float height = (face.bbox.lt.y - face.bbox.rb.y);

            float xWorld = x * matToTextureCoordinatesX;
            float yWorld = y * matToTextureCoordinatesY;

            float widthWorld = width * matToTextureCoordinatesX;
            float heightWorld = height * matToTextureCoordinatesY;

            //Create a UI Game object from a prefab
            GameObject faceRect = Instantiate(Resources.Load("Prefabs/FaceUI")) as GameObject;
            faceRect.tag = "Face";
            faceRect.layer = LayerMask.NameToLayer("UI");
            faceRect.transform.SetParent(UI.transform, false);
            RectTransform rt = faceRect.GetComponent<RectTransform>();

            rt.anchoredPosition = new Vector2(xWorld - widthWorld/2, yWorld - heightWorld/2);
            rt.sizeDelta = new Vector2(widthWorld, heightWorld);
            rt.anchorMin = new Vector2(0, 0);
            rt.anchorMax = new Vector2(0, 0);
            rt.pivot = new Vector2(0, 0);
            rt.localScale = new Vector3(1, 1, 1);

            // Replace Text game object in prefab with the emotion text
            TextMeshProUGUI emotionText = faceRect.transform.Find("EmotionText").GetComponent<TextMeshProUGUI>();
            if (face.emotion != null)
            {
                emotionText.text = face.emotion.GetEmotionText();
            }
            else
            {
                emotionText.text = "-";
            }

            /*
            GameObject faceRect = new GameObject("FaceRect");
            faceRect.layer = LayerMask.NameToLayer("UI");
            faceRect.AddComponent<CanvasRenderer>();
            faceRect.transform.SetParent(UI.transform, false);

            faceRect.tag = "Face";
            

            RawImage faceImage = faceRect.AddComponent<RawImage>();
            Texture2D texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, UnityEngine.Color.red);
            texture.Apply();
            faceImage.texture = texture;
            */

            // Draw the emotion over the face
            // if (face.emotion != null)
            // {
            //     // draw the emotion icon
            //     int emotionId = face.emotion.GetEmotion();
            //     Sprite emotionImage = emotionImages[emotionId];
            //     faceImage.sprite = emotionImage;
            // }

            
            //Construct a rectangle from the bounding box of face
            /*
            Rectangle rect = new Rectangle((int)(x-width/2), (int)(y-height/2), (int)width, (int)height);
            Mat matImage = ImageUtils.ConvertTextureToMat(outputImage.texture as Texture2D, DepthType.Cv8U, 8, 4);
            CvInvoke.Rectangle(matImage, rect, new MCvScalar(0, 255, 0), 2);

            //Print the landmakrs of the face
            foreach (Point2D landmark in face.landmark.points)
            {
                CvInvoke.Circle(matImage, new System.Drawing.Point((int)landmark.x, (int)landmark.y), 2, new MCvScalar(0, 0, 255), 2);
            }

            //Draw the emotion over the face
            // Get the icon png from /Assets/Resources/EmotionImages load image


            // if (face.emotion != null)
            // {
            //     // draw the emotion icon
            //     int emotionId = face.emotion.GetEmotion();
            //     Mat emotionImage = emotionImages[emotionId];

            //     // Resize emotionImage to have the same width - position as the original image
            //     CvInvoke.Resize(emotionImage, emotionImage, new Size((int)(width - x), (int)height));

            //     string emotionImageInfo = "emotionImage: " + emotionImage.Size;
            //     Debug.Log(emotionImageInfo);
            //     //CvInvoke.PutText(matImage, face.emotion.GetEmotionText(), new Point((int)x, (int)(y) - 10), FontFace.HersheySimplex, -1, new MCvScalar(0, 0, 255), 2);
            // }
            
            Texture2D texture = ImageUtils.ConvertMatToTexture(matImage);
            outputImage.texture = texture;
            */
        }

        

        // Print mat dimensions
        //Debug.Log(testImage.Size);

        //outputImage.texture = webcam.texture;
        //
        //Debug.Log(matImage.Size);
        //faceDetector.Detect(matImage);
    }
}