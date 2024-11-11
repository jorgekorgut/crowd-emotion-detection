using UnityEngine;

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
}