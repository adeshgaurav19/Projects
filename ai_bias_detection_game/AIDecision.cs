using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.Networking;

public class AIDecision : MonoBehaviour
{
    public Text scenarioText;
    public Text decisionText;
    public Text biasText;
    public Text statusText;
    public GameObject drone;

    // ✅ Different Environments in Unity
    public GameObject cityEnvironment;
    public GameObject desertEnvironment;
    public GameObject battlefieldEnvironment;
    public GameObject borderEnvironment;
    public GameObject mallEnvironment;

    private string apiUrl = "http://127.0.0.1:5000/ai_decision"; // Flask API URL

    public void GetAIDecision()
    {
        StartCoroutine(RequestAIDecision());
    }

    IEnumerator RequestAIDecision()
    {
        UnityWebRequest request = UnityWebRequest.Get(apiUrl);
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonResult = request.downloadHandler.text;
            AIDecisionResponse data = JsonUtility.FromJson<AIDecisionResponse>(jsonResult);

            // ✅ Update UI Elements
            scenarioText.text = "Scenario: " + data.scenario;
            decisionText.text = "AI Decision: " + data.decision;
            biasText.text = "Bias Score: " + data.bias_score.ToString("F2");

            // ✅ Set Environment Based on AI Scenario
            SetEnvironment(data.environment);

            if (data.biased)
            {
                statusText.text = "🚨 Bias Detected! Drone Auto-Killed!";
                statusText.color = Color.red;
                StartCoroutine(DestroyDrone());
            }
            else
            {
                statusText.text = "✅ Decision Passed.";
                statusText.color = Color.green;
                MoveDroneUp();
            }
        }
        else
        {
            statusText.text = "⚠️ AI Decision Error!";
            statusText.color = Color.yellow;
        }
    }

    void SetEnvironment(string environment)
    {
        // ❌ Disable all environments
        cityEnvironment.SetActive(false);
        desertEnvironment.SetActive(false);
        battlefieldEnvironment.SetActive(false);
        borderEnvironment.SetActive(false);
        mallEnvironment.SetActive(false);

        // ✅ Enable the correct environment
        switch (environment)
        {
            case "City":
                cityEnvironment.SetActive(true);
                break;
            case "Desert":
                desertEnvironment.SetActive(true);
                break;
            case "Battlefield":
                battlefieldEnvironment.SetActive(true);
                break;
            case "Border":
                borderEnvironment.SetActive(true);
                break;
            case "Mall":
                mallEnvironment.SetActive(true);
                break;
        }
    }

    IEnumerator DestroyDrone()
    {
        yield return new WaitForSeconds(1);
        Destroy(drone); // 💥 Remove drone from scene
    }

    void MoveDroneUp()
    {
        if (drone != null)
        {
            drone.transform.position += new Vector3(0, 1, 0); // ⬆ Move up slightly
        }
    }
}

[System.Serializable]
public class AIDecisionResponse
{
    public string scenario;
    public string decision;
    public float bias_score;
    public bool biased;
    public string environment;
}
