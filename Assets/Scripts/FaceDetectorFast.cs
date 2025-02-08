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

namespace FaceDetectorFast
{
    class Point2D
    {
        public float x, y;
        public Point2D(float x, float y)
        {
            this.x = x;
            this.y = y;
        }
    }

    class BBox
    {
        public Point2D lt, rb;

        public BBox(Point2D lt, Point2D rb)
        {
            this.lt = lt;
            this.rb = rb;
        }
    }

    class Landmark
    {
        public Point2D[] points;
        public Landmark(Point2D[] points)
        {
            this.points = points;
        }
    }

    class Face
    {
        public BBox bbox;
        public Landmark landmark;
        public float confidence;
        public Emotion emotion = null;
        public Face(BBox bbox, Landmark landmark, float confidence)
        {
            this.bbox = bbox;
            this.landmark = landmark;
            this.confidence = confidence;
        }

        public void ToOriginalCoordinates(int paddingX, int paddingY, float modelToOriginalRatioX, float modelToOriginalRatioY, int imageInputWidth, int imageInputHeight)
        {
            float xMin = Math.Max((bbox.lt.x - paddingX) * modelToOriginalRatioX, 0);
            float yMin = Math.Max((bbox.lt.y - paddingY) * modelToOriginalRatioY, 0);
            float xMax = Math.Min((bbox.rb.x + paddingX) * modelToOriginalRatioX, imageInputWidth - 1);
            float yMax = Math.Min((bbox.rb.y + paddingY) * modelToOriginalRatioY, imageInputHeight - 1);

            bbox.lt.x = xMin;
            bbox.lt.y = imageInputHeight - yMin;
            bbox.rb.x = xMax;
            bbox.rb.y = imageInputHeight - yMax; //Invert y axis for Unity coordinates

            foreach (Point2D point in landmark.points)
            {
                point.x = (point.x + paddingX) * modelToOriginalRatioX;
                point.y = (point.y + paddingY) * modelToOriginalRatioY;

                point.y = imageInputHeight - point.y; //Invert y axis for Unity coordinates
            }
        }

        public override string ToString()
        {
            string faceString = $"Face: {bbox.lt.x}, {bbox.lt.y}, {bbox.rb.x}, {bbox.rb.y}\n";
            faceString += $"Confidence: {confidence}\n";
            foreach (Point2D point in landmark.points)
            {
                faceString += $"Landmark: {point.x}, {point.y}\n";
            }
            
            if(emotion != null)
            {
                faceString += $"Emotion: {emotion.GetEmotionText()}\n";
            }

            return faceString;
        }

    }

    class Filter
    {
        /*
            * Intersection over Union (Jaccardi`s index).
            * The intersection area divided by the union area of two bounding boxes.
            */
        public static float IntersectionOverUnion(Face face1, Face face2)
        {
            float x1 = Mathf.Max(face1.bbox.lt.x, face2.bbox.lt.x);
            float y1 = Mathf.Max(face1.bbox.lt.y, face2.bbox.lt.y);
            float x2 = Mathf.Min(face1.bbox.rb.x, face2.bbox.rb.x);
            float y2 = Mathf.Min(face1.bbox.rb.y, face2.bbox.rb.y);
            float intersection = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
            float area1 = (face1.bbox.rb.x - face1.bbox.lt.x) * (face1.bbox.rb.y - face1.bbox.lt.y);
            float area2 = (face2.bbox.rb.x - face2.bbox.lt.x) * (face2.bbox.rb.y - face2.bbox.lt.y);

            return intersection / (area1 + area2 - intersection);
        }

        /*
        * Non-Maximum Suppression.
        * Remove the bounding boxes with high intersection over union.
        */
        public static List<Face> NonMaximumSuppression(List<Face> faces, float threshold)
        {
            List<Face> result = new List<Face>();
            faces.Sort((a, b) => b.confidence.CompareTo(a.confidence));

            foreach (Face face in faces)
            {
                bool keep = true;
                foreach (Face other in result)
                {
                    if (IntersectionOverUnion(face, other) > threshold)
                    {
                        keep = false;
                        break;
                    }
                }
                if (keep)
                {
                    result.Add(face);
                }
            }
            return result;
        }
    }

    class FaceDetector
    {
        private int modelNumberOfChannels = 3;
        private int modelImageWidth = 640;
        private int modelImageHeight = 640;
        private float confThreshold;
        private float nonMaxSuppressionThreshold;
        private bool isLoaded = false;
        private Net net;

        private int paddingX = 20;
        private int paddingY = 20;
        public FaceDetector(string modelpath, float confThreshold, float nmsThreshold)
        {
            this.confThreshold = confThreshold;
            this.nonMaxSuppressionThreshold = nmsThreshold;

            readNet(modelpath);
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

        public List<Face> Detect(Mat imageInput)
        {
            if (!isLoaded)
            {
                Debug.Log("Model is not loaded.");
                return new List<Face>();
            }

            Mat preprocessedFrame = Preprocess(imageInput);

            VectorOfMat netOutput = new VectorOfMat();
            net.SetInput(preprocessedFrame);
            net.Forward(netOutput, net.UnconnectedOutLayersNames);

            List<Face> faces = Postprocess(netOutput, imageInput.Width, imageInput.Height);

            //Debug.Log("faces Count : " + faces.Count);  
            return faces;
            //return new List<Face>();
        }

        private Mat Preprocess(Mat img)
        {
            Mat rgbImage = new Mat(new Size(modelImageWidth, modelImageHeight), img.Depth, modelNumberOfChannels);
            CvInvoke.CvtColor(img, rgbImage, ColorConversion.Rgba2Rgb);
            CvInvoke.Flip(rgbImage, rgbImage, FlipType.Vertical); // Flip image because the model and unity have different coordinate systems (y = -y)

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

        private List<Face> Postprocess(VectorOfMat modelOutput, int imageInputWidth, int imageInputHeight)
        {
            float modelToOriginalRatioX = (float)imageInputWidth / modelImageWidth;
            float modelToOriginalRatioY = (float)imageInputHeight / modelImageHeight;

            Mat mat = modelOutput[0];

            List<Face> faces = CreatePropositionsFaceArray(mat);

            List<Face> nonMaxSuppressedFaces = Filter.NonMaximumSuppression(faces, nonMaxSuppressionThreshold);

            foreach (Face currentFace in nonMaxSuppressedFaces)
            {
                currentFace.ToOriginalCoordinates(paddingX, paddingY, modelToOriginalRatioX, modelToOriginalRatioY, imageInputWidth, imageInputHeight);
            }

            return nonMaxSuppressedFaces;
        }

        private List<Face> CreatePropositionsFaceArray(Mat mat)
        {
            List<Face> faces = new List<Face>();
            Array data = mat.GetData();

            int[] dimensions = mat.SizeOfDimension;
            int numberFeatures = dimensions[1];
            int numberEvaluatedPixels = dimensions[2];

            for (int currentEvaluatedPixel = 0; currentEvaluatedPixel < numberEvaluatedPixels; currentEvaluatedPixel++)
            {
                float maxConfidence = (float)data.GetValue(0, 4, currentEvaluatedPixel);

                if (maxConfidence > confThreshold)
                {
                    float x = (float)data.GetValue(0, 0, currentEvaluatedPixel);
                    float y = (float)data.GetValue(0, 1, currentEvaluatedPixel);
                    float width = (float)data.GetValue(0, 2, currentEvaluatedPixel);
                    float height = (float)data.GetValue(0, 3, currentEvaluatedPixel);

                    float xMin = x - (0.5f * width);
                    float yMin = y - (0.5f * height);
                    float xMax = x + (0.5f * width);
                    float yMax = y + (0.5f * height);

                    Point2D[] landMarks = new Point2D[0];

                    Face face = new Face(
                        new BBox(new Point2D(xMin, yMin), new Point2D(xMax, yMax)),
                        new Landmark(landMarks),
                        maxConfidence
                    );

                    faces.Add(face);
                }
            }
            return faces;
        }
    }
}
