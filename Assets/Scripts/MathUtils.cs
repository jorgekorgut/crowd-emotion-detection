class MathUtils
{

    public float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    public float[] Softmax(float[] x)
    {
        float[] result = new float[x.Length];
        float max = x.Max();
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