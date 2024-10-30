using UnityEngine;

using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class HelloWord : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        String win1 = "Test Window"; //The name of the window
        CvInvoke.NamedWindow(win1); //Create the window using the specific name

        Mat img = new Mat(200, 400, DepthType.Cv8U, 3); //Create a 3 channel image of 400x200
        img.SetTo(new Bgr(255, 0, 0).MCvScalar); // set it to Blue color

        //Draw "Hello, world." on the image using the specific font
        CvInvoke.PutText(
           img,
           "Hello, world",
           new System.Drawing.Point(10, 80),
           FontFace.HersheyComplex,
           1.0,
           new Bgr(0, 255, 0).MCvScalar);


        CvInvoke.Imshow(win1, img); //Show the image
        CvInvoke.WaitKey(0);  //Wait for the key pressing event
        CvInvoke.DestroyWindow(win1); //Destroy the window if key is pressed
    }

    // Update is called once per frame
    void Update()
    {

    }
}
