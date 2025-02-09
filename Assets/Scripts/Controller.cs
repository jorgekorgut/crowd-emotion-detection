using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Drawing;
using TMPro;
using System.Threading.Tasks;

using UnityEngine.Networking;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.IO;

using FaceDetectorFast;

public class Controller : MonoBehaviour
{

    public RawImage outputImage;

    [SerializeField] private TMP_Dropdown dropdown;

    GameObject UIController;

    UIEmotionScore UIEmotionScoreController;

    UnityEngine.UI.Image UIEmotionPredominantImage;

    public GameObject UI;
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

    private bool isFaceTrackerActive = false;

    private bool isEmotionScoreActive = true;

    private bool isPredominantEmotionActive = false;

    void Start()
    {
        Debug.developerConsoleVisible = true;

        this.webcam = new Webcam();
        if(this.webcam.isLoaded)
        {
            outputImage.texture = this.webcam.texture;
        }
        else
        {
            Debug.Log("Webcam not loaded, Loading image from file instead.");
            //string filename = "Assets/Resources/Images/femme.jpg";
            //string filename = "Assets/Resources/Images/Student_in_Class_Tulane_University_September_2002.jpg"; // author: https://www.flickr.com/people/28035080@N04
            //string filename = "Assets/Resources/Images/lots-of-students.jpg"; 
            //string filename = "Assets/Resources/Images/smiley-class.jpg"; 
            string filename = "Images/smiley-class.jpg"; 

            filename = Path.Combine(Application.streamingAssetsPath, filename);
            var rawData = System.IO.File.ReadAllBytes(filename);
            Texture2D texture = new Texture2D(2, 2);
            texture.LoadImage(rawData);
            outputImage.texture = texture;
        }

        
        this.faceDetector = new FaceDetector("FaceDetection/yolov11n-face", 0.45f, 0.5f);
        
        this.emotionDetector = new EmotionDetector("EmotionDetection/mobilevit_va_mtl");

        // Load the emotions images
        emotionImages = FileUtils.LoadEmotionsImages();

        matToTextureCoordinatesX = (float)outputImage.rectTransform.rect.width / (float)outputImage.texture.width;
        matToTextureCoordinatesY = (float)outputImage.rectTransform.rect.height / (float)outputImage.texture.height;

        drawUI();
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
            UpdateUI();
            lastTimeUpdateDetection = (int)(Time.time * 1000);
        }

        ShowHideUI();
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

        if(isFaceTrackerActive)
        {
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
                emotionText.outlineColor = new Color32(50, 80, 100, 255);
                emotionText.outlineWidth = 0.25f;
                if (face.emotion != null)
                {
                    emotionText.text = face.emotion.GetEmotionText();
                }
                else
                {
                    emotionText.text = "-";
                }
            }
        }
    }

    private void ShowHideUI()
    {
        if(isEmotionScoreActive)
        {
            UIEmotionScoreController.gameObject.SetActive(true);
        }
        else
        {
            UIEmotionScoreController.gameObject.SetActive(false);
        }

        if(isPredominantEmotionActive)
        {
            UIEmotionPredominantImage.gameObject.transform.parent.gameObject.SetActive(true);
        }
        else
        {
            UIEmotionPredominantImage.gameObject.transform.parent.gameObject.SetActive(false);
        }

        if(!isFaceTrackerActive)
        {
            GameObject[] facesGameObject = GameObject.FindGameObjectsWithTag("Face");
            foreach (GameObject face in facesGameObject)
            {
                Destroy(face);
            }
        }
    }

    private void drawUI()
    {
        UIController = Instantiate(Resources.Load("Prefabs/UI")) as GameObject;
        UIController.tag = "UI";
        UIController.layer = LayerMask.NameToLayer("UI");
        UIController.transform.SetParent(UI.transform, false);

        // Set UI Dropdown Options
        dropdown = UIController.transform.Find("Dropdown").GetComponent<TMP_Dropdown>();

        int options = 0;
        options |= isFaceTrackerActive ? 1 : 0;
        options |= isEmotionScoreActive ? 1<<1 : 0;
        //options |= isFaceTrackerActive ? 1<<1 : 0;
        dropdown.value = options;

        dropdown.onValueChanged.AddListener(delegate {
            UpdateDropDownValues(dropdown.value);
        });

        UIEmotionScoreController = UIController.transform.Find("EmotionEscore").GetComponent<UIEmotionScore>();

        UIEmotionPredominantImage = UIController.transform.Find("EmotionPredominant").Find("EmotionImage").GetComponent<UnityEngine.UI.Image>();

        UIController.transform.Find("ChangeCamera").GetComponent<Button>().onClick.AddListener(() =>
        {
            outputImage.texture = webcam.SwitchCamera();
            matToTextureCoordinatesX = (float)outputImage.rectTransform.rect.width / (float)outputImage.texture.width;
            matToTextureCoordinatesY = (float)outputImage.rectTransform.rect.height / (float)outputImage.texture.height;
        });
    }

    private void UpdateUI()
    {
        // Update the UI with the emotion score
        if(faces != null)
        {
            float score = 0f;

            int count = 0;
            foreach (Face face in faces)
            {
                if (face.emotion != null)
                {
                    count++;
                    score += face.emotion.GetEmotionScore();
                }
            }

            if(count == 0)
            {
                score = 0;
            }
            else
            {
                score/=count;
            }
            UIEmotionScoreController.SetScore(score);
        }

        // Update the UI with the predominant emotion
        if(faces != null)
        {
            int[] emotionsCount = new int[Emotion.emotions.Length];
            foreach (Face face in faces)
            {
                if (face.emotion != null)
                {
                    emotionsCount[face.emotion.GetEmotion()]++;
                }
            }

            int predominantEmotion = 5;
            int maxCount = 0;
            for (int i = 0; i < emotionsCount.Length; i++)
            {
                if (emotionsCount[i] > maxCount)
                {
                    maxCount = emotionsCount[i];
                    predominantEmotion = i;
                }
            }

            UIEmotionPredominantImage.sprite = emotionImages[predominantEmotion];
        }
        
    }

    public void UpdateDropDownValues(int options)
    {
        Debug.Log("Selected option: " + options);

        // 0: Face and Emotion Tracker
        isFaceTrackerActive = (options & 1) == 1;

        // 1: Emotion Score
        isEmotionScoreActive = (options>>1 & 1) == 1;

        // 2: Predominant Emotion
        isPredominantEmotionActive = (options>>2 & 1) == 1;

        //isFaceTrackerActive = (options>>1 & 1) == 1;
    }
}