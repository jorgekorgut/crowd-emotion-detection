using UnityEngine;
using System.Collections.Generic;

class MathUtils
{
    public static float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    public static float[] Softmax(float[] x)
    {
        float[] result = new float[x.Length];
        float max = Mathf.Max(x);
        float sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = Mathf.Exp(x[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < x.Length; i++)
        {
            result[i] /= sum;
        }
        return result;
    }

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