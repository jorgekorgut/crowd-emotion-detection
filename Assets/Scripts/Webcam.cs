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

    public bool isLoaded = false;

    private int deviceId;

    public Webcam()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.Log("No camera detected");
            //camAvailable = false;
            return;
        }

        deviceId = 0;

        texture = new WebCamTexture(devices[deviceId].name, Screen.width, Screen.height);
        if (texture == null)
        {
            Debug.Log("Unable to open camera");
            return;
        }

        isFrontCam = devices[deviceId].isFrontFacing;
        texture.Play();
        isLoaded = true;
        //camAvailable = true;
    }

    public WebCamTexture SwitchCamera()
    {
        // Change camera to frontal or back
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.Log("No camera detected");
            //camAvailable = false;
            return null;
        }

        if(devices.Length == 1)
        {
            Debug.Log("Only one camera detected");
            return null;
        }

        deviceId = (deviceId + 1) % devices.Length;

        texture.Stop();
        texture = new WebCamTexture(devices[deviceId].name, Screen.width, Screen.height);
        if (texture == null)
        {
            Debug.Log("Unable to open camera");
            return null;
        }

        isFrontCam = devices[deviceId].isFrontFacing;
        texture.Play();
        return texture;
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
