using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Webcam
{
    public WebCamTexture texture;

    //private bool camAvailable = false;

    private bool isFrontCam = false;

    public Webcam()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.Log("No camera detected");
            //camAvailable = false;
            return;
        }
        texture = new WebCamTexture(devices[0].name, Screen.width, Screen.height);
        if (texture == null)
        {
            Debug.Log("Unable to open camera");
            return;
        }

        isFrontCam = devices[0].isFrontFacing;
        texture.Play();
        //camAvailable = true;
    }

    public void Update()
    {
        // if (camAvailable)
        // {
        //     float ratio = (float)camTexture.width / camTexture.height;

        //     float scaleX = isFrontCam ? -1f : 1f;
        //     float scaleY = camTexture.videoVerticallyMirrored ? -1f : 1f;
        //     rawImage.rectTransform.localScale = new Vector3(scaleX, scaleY, 1f);

        //     int orient = -camTexture.videoRotationAngle;
        //     rawImage.rectTransform.localEulerAngles = new Vector3(0, 0, orient);
        // }
    }
}
