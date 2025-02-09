using UnityEngine;
using UnityEngine.UI;
public class UIEmotionScore : MonoBehaviour
{
    public GameObject positiveMask;
    public GameObject negativeMask;

    // Score is a float between -1 and 1
    public float Score = 0;
    void Start()
    {
        // shrink positive mask to be 0
        positiveMask.transform.localScale = new Vector3(0f, 1, 1);
        negativeMask.transform.localScale = new Vector3(0f, 1, 1);
    }

    public void SetScore(float score)
    {
        this.Score = score;

        if(score > 0)
        {
            positiveMask.transform.localScale = new Vector3(score, 1, 1);
            negativeMask.transform.localScale = new Vector3(0f, 1, 1);
        }
        else
        {
            positiveMask.transform.localScale = new Vector3(0f, 1, 1);
            negativeMask.transform.localScale = new Vector3(-score, 1, 1);
        }
    }
}
