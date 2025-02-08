using System;
using UnityEngine;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;

public static class ImageUtils
{

    public static Mat CropImage(Mat image, Rectangle cropRect)
    {
        Mat copy = image.Clone();

        Rectangle cropHeight = new Rectangle(0, cropRect.Y, image.Width, cropRect.Height);

        Mat imageCroppedHeight = new Mat(copy, cropHeight);

        Mat transposedImage = new Mat(imageCroppedHeight.Height, imageCroppedHeight.Width, imageCroppedHeight.Depth, imageCroppedHeight.NumberOfChannels);

        CvInvoke.Transpose(imageCroppedHeight, transposedImage);

        Rectangle cropWidth = new Rectangle(0, cropRect.X, imageCroppedHeight.Height, cropRect.Width);

        Mat imageCroppedWidth = new Mat(transposedImage, cropWidth);
        
        Mat finalImage = new Mat(cropRect.Height, cropRect.Width, imageCroppedWidth.Depth, imageCroppedWidth.NumberOfChannels);

        CvInvoke.Transpose(imageCroppedWidth, finalImage);

        return finalImage;
    }
    
    public static Mat ConvertTextureToMat(Texture2D frame, DepthType depthType, int nBytePerPixel, int channels)
    {
        if (frame == null)
        {
            return null;
        }

        //var frameData = new Color32[frame.width * frame.height];
        Color32[] frameData = frame.GetPixels32();
        return ConvertTextureToMat(frameData,frame.width, frame.height, depthType, nBytePerPixel, channels);
    }

    public static Mat ConvertTextureToMat(WebCamTexture frame, DepthType depthType, int nBytePerPixel, int channels)
    {
        if (frame == null)
        {
            return null;
        }

        //var frameData = new Color32[frame.width * frame.height];
        Color32[] frameData = frame.GetPixels32();
        return ConvertTextureToMat(frameData,frame.width,frame.height,  depthType, nBytePerPixel, channels);
    }

    public static Mat ConvertTextureToMat(Color32[] frameData, int width, int height, DepthType depthType, int nBytePerPixel, int channels)
    {
        /*
            Mat mat = new Mat(frame.height, frame.width, depthType, channels);
            Color32[] pixels = frame.GetPixels32();
            byte[] data = new byte[pixels.Length * nBytePerPixel];
            for (int i = 0; i < pixels.Length; i++)
            {
                data[i * nBytePerPixel] = pixels[i].r;
                data[i * nBytePerPixel + 1] = pixels[i].g;
                data[i * nBytePerPixel + 2] = pixels[i].b;
                if (nBytePerPixel == 4)
                {
                    data[i * nBytePerPixel + 3] = pixels[i].a;
                }
            }
            mat.SetTo(data);
            return mat;
        */
        Image<Rgba, Byte> img = new Image<Rgba, byte>(width, height);

        try
        {
            
            // TODO: Slow, find a better way.
            for (int r = 0; r < height; ++r)
            {
                for (int c = 0; c < width; ++c)
                {
                    var pixel = frameData[r * width + c];
                    img[r, c] = new Rgba(pixel.r, pixel.g, pixel.b, pixel.a);
                }
            }

            //System.Runtime.InteropServices.Marshal.Copy(sourceMat.DataPointer, imageData, 0, imageData.Length);

            return img.Mat;
        }
        catch (Exception e)
        {
            Debug.Log(e.Message);
        }
        return null;
    }

    public static Mat LoadJPGToMat(string path)
    {
        Mat img = CvInvoke.Imread(path, ImreadModes.Color);
        //invert r and b channels
        CvInvoke.CvtColor(img, img, ColorConversion.Bgr2Rgb);
        //flip image y
        CvInvoke.Flip(img, img, FlipType.Vertical);
        return img;
    }

    public static Texture2D ConvertMatToTexture(Mat sourceMat)
    {
        //Get the height and width of the Mat 
        int imgHeight = sourceMat.Height;
        int imgWidth = sourceMat.Width;
        int imgChannels = sourceMat.NumberOfChannels;
        TextureFormat format = TextureFormat.RGBA32;

        if (imgChannels == 3)
        {
            format = TextureFormat.RGB24;
        }

        Texture2D texture = new Texture2D(imgWidth, imgHeight, format, false);
        byte[] imageData = new byte[imgHeight * imgWidth * imgChannels];
        System.Runtime.InteropServices.Marshal.Copy(sourceMat.DataPointer, imageData, 0, imageData.Length);

        texture.LoadRawTextureData(imageData);
        texture.Apply();

        return texture;
    }
}