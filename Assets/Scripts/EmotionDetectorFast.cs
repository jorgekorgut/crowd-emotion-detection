using UnityEngine;
using System;

using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Util;

using System.Drawing;
using System.Collections.Generic;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Linq;
using System.Runtime.InteropServices;
using System.IO;

public class Emotion
{
    public static string[] emotions = new string[] { "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise" };

    private float[] emotionScores;

    private float valence;

    private float arousal;
    public Emotion(float[] scores)
    {
        emotionScores = new float[8];
        for (int i = 0; i < emotionScores.Length; ++i)
        {
            this.emotionScores[i] = scores[i];
        }

        this.emotionScores = MathUtils.Softmax(emotionScores);
        this.valence = (float)Math.Tanh(scores[8]);
        this.arousal = (float)Math.Tanh(scores[9]);
    }

    public int GetEmotion()
    {
        int bestInd = 0;
        if (emotionScores != null)
        {
            float maxScore = 0;
            for (int i = 0; i < emotionScores.Length; ++i)
            {
                if (maxScore < emotionScores[i])
                {
                    maxScore = emotionScores[i];
                    bestInd = i;
                }
            }
        }

        return bestInd;
    }
    public string GetEmotionText()
    {
        return emotions[GetEmotion()];
    }

    public float GetValence()
    {
        return valence;
    }

    public float GetArousal()
    {
        return arousal;
    }
}

public class EmotionDetector
{
    private int modelNumberOfChannels = 3;
    private int modelImageWidth = 224;
    private int modelImageHeight = 224;

    private bool isLoaded = false;

    private Net net;

    public EmotionDetector(string modelPath)
    {
        readNet(modelPath);
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

    public Emotion Detect(Mat imageInput)
    {
        //Debug.Log("Detecting Emotion...");
        if (!isLoaded)
        {
            Debug.Log("Model is not loaded.");
            //return ;
        }

        Mat preprocessedFrame = Preprocess(imageInput);

        VectorOfMat netOutput = new VectorOfMat();
        net.SetInput(preprocessedFrame);
        net.Forward(netOutput, net.UnconnectedOutLayersNames);

        Emotion emotion = Postprocess(netOutput);

        return emotion;
    }

    private Mat Preprocess(Mat image)
    {
        Mat rgbImage = new Mat(new Size(modelImageWidth, modelImageHeight), image.Depth, modelNumberOfChannels);
        CvInvoke.CvtColor(image, rgbImage, ColorConversion.Rgba2Rgb);
        CvInvoke.Flip(rgbImage, rgbImage, FlipType.Vertical);

        Mat inputBlob = DnnInvoke.BlobFromImage(
            rgbImage,
            1.0 / 255,
            new Size(modelImageWidth, modelImageHeight),
            new MCvScalar(0, 0, 0),
            false, // SwapRB
            false // Crop
        );

        return inputBlob;
    }

    private Emotion Postprocess(VectorOfMat netOutput)
    {
        Mat output = netOutput[0];
        float[] emotionScores = new float[output.SizeOfDimension[1]];
        Marshal.Copy(output.DataPointer, emotionScores, 0, output.SizeOfDimension[1]);

        string emotionInfo = "emotionScores: ";
        for (int i = 0; i < emotionScores.Length; ++i)
        {
            emotionInfo += emotionScores[i] + " ";
        }

        Emotion emotion = new Emotion(emotionScores);
        return emotion;
    }
}