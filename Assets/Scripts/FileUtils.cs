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
            string filename = Path.Combine(Application.streamingAssetsPath, "EmotionImages", Emotion.emotions[i] + ".png");

            var rawData = System.IO.File.ReadAllBytes(filename);
            Texture2D outputImage = new Texture2D(2, 2);
            outputImage.LoadImage(rawData);
            Sprite sprite = Sprite.Create(outputImage, new Rect(0, 0, outputImage.width, outputImage.height), new Vector2(0.5f, 0.5f));
            emotionImages[i] = sprite;
        }

        return emotionImages;
    }
}