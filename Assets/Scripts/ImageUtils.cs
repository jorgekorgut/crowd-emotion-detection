using System;
using UnityEngine;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public static class ImageUtils
{
    public static Mat ConvertWebCamTextureToMat(WebCamTexture frame, DepthType depthType, int nBytePerPixel, int channels)
    {
        try
        {
            Image<Rgba, Byte> img = new Image<Rgba, byte>(frame.width, frame.height);
            var frameData = new Color32[frame.width * frame.height];
            frame.GetPixels32(frameData);

            // TODO: Slow, find a better way.
            for (int r = 0; r < frame.height; ++r)
            {
                for (int c = 0; c < frame.width; ++c)
                {
                    var pixel = frameData[r * frame.width + c];
                    img[r, c] = new Rgba(pixel.r, pixel.g, pixel.b, pixel.a);
                }
            }

            return img.Mat;
        }
        catch (Exception e)
        {
            Debug.Log(e.Message);
        }
        return null;
    }

    public static Texture2D ConvertMatToTexture(Mat sourceMat)
    {
        //Get the height and width of the Mat 
        int imgHeight = sourceMat.Height;
        int imgWidth = sourceMat.Width;
        int imgChannels = sourceMat.NumberOfChannels;

        Texture2D texture = new Texture2D(imgWidth, imgHeight, TextureFormat.RGBA32, false);
        byte[] imageData = new byte[imgHeight * imgWidth * imgChannels];
        System.Runtime.InteropServices.Marshal.Copy(sourceMat.DataPointer, imageData, 0, imageData.Length);
        
        texture.LoadRawTextureData(imageData);
        texture.Apply();

        return texture;
    }
}