using System;
using UnityEngine;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public static class ImageUtils {

    public static Mat ConvertWebCamTextureToMat(WebCamTexture frame, DepthType depthType, int nBytePerPixel, int channels) {
        try {
            Image<Bgra, Byte> img = new Image<Bgra, byte>(frame.width, frame.height);
            var frameData = new Color32[frame.width * frame.height];
            frame.GetPixels32(frameData);

            // TODO: Slow, find a better way.
            for (int r = 0; r < frame.height; ++r) {
                for (int c = 0; c < frame.width; ++c) {
                    var pixel = frameData[r * frame.width + c];
                    img[r, c] = new Bgra(pixel.b, pixel.g, pixel.r, pixel.a);
                }
            }

            return img.Mat;
        } catch (Exception e) {
            Debug.Log(e.Message);
        }
        return null;
    }
}