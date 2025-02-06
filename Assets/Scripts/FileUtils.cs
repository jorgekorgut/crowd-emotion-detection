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

public static class FileUtils{
    
    public static void PrintToCSV(Mat mat, string name)
    {
        Array data = mat.GetData();

        for (int k = 0; k < data.GetLength(1); k++)
        {
            using (StreamWriter writer = new StreamWriter($"Assets/Resources/Matrix/{name}_{k}.csv"))
            {
                for (int i = 0; i < data.GetLength(2); i++)
                {
                    for (int j = 0; j < data.GetLength(3); j++)
                    {
                        writer.Write($"{data.GetValue(0, k, i, j)}");
                        if (j < data.GetLength(3) - 1)
                        {
                            writer.Write(",");
                        }
                    }
                    writer.WriteLine();
                }
            }
        }
    }

    public static Sprite[] LoadEmotionsImages()
    {
        Sprite[] emotionImages = new Sprite[8];
        for (int i = 0; i < 8; i++)
        {
            emotionImages[i] = Resources.Load<Sprite>($"EmotionImages/{Emotion.emotions[i]}");
        }

        return emotionImages;

        /*
        Mat[] emotionImages = new Mat[8];
        for (int i = 0; i < 8; i++)
        {
            emotionImages[i] = ImageUtils.LoadJPGToMat($"Assets/Resources/EmotionImages/{Emotion.emotions[i]}.png");
        }

        // Resize image to be 48x48
        for (int i = 0; i < 8; i++)
        {
            CvInvoke.Resize(emotionImages[i], emotionImages[i], new Size(48, 48));
        }

        return emotionImages;
        */
    }
}