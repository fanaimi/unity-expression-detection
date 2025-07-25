using System.Collections;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

public class EmotionRecogniser : MonoBehaviour
{
    public bool debug = false;
    [Header("UI & Model Settings")]
    public RawImage rawImage;
    public NNModel emotionModelAsset;
    public Text emotionTextUI;

    private Model runtimeModel;
    private IWorker worker;
    private WebCamTexture webcamTexture;

    private readonly string[] emotionLabels = {
        "neutral", "happiness", "surprise", "sadness",
        "anger", "disgust", "fear", "contempt"
    };

    private float interval = 1.0f; // inference interval in seconds

    IEnumerator Start()
    {
        runtimeModel = ModelLoader.Load(emotionModelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);

        // Wait for webcam to be ready
        while (rawImage.texture == null || !(rawImage.texture is WebCamTexture))
        {
            Debug.Log("Waiting for webcam texture...");
            yield return null;
        }

        webcamTexture = rawImage.texture as WebCamTexture;

        while (!webcamTexture.isPlaying)
        {
            Debug.Log("Waiting for webcam to start...");
            yield return null;
        }

        Debug.Log("Webcam ready. Starting emotion recognition.");
        StartCoroutine(AnalyzeRoutine());
    }

    IEnumerator AnalyzeRoutine()
    {
        while (true)
        {
            RecognizeEmotionFromWebcam();
            yield return new WaitForSeconds(interval);
        }
    }

    void RecognizeEmotionFromWebcam()
    {
        // 1. Capture frame
        Texture2D snap = new Texture2D(webcamTexture.width, webcamTexture.height);
        snap.SetPixels32(webcamTexture.GetPixels32());
        snap.Apply();

        // 2. Resize to 64x64
        Texture2D resized = ResizeTo64x64(snap);
        Color[] pixels = resized.GetPixels();


        // 3. Preprocess: grayscale + normalize to [-1, 1]
        float[] input = new float[64 * 64];
        for (int i = 0; i < pixels.Length; i++)
        {
            float gray = pixels[i].r * 0.299f + pixels[i].g * 0.587f + pixels[i].b * 0.114f;
            input[i] = (gray - 0.5f) * 2f;  // normalize to [-1, 1]
        }

        // 4. Create tensor in NHWC format: (1, 64, 64, 1)
        Tensor tensor = new Tensor(1, 64, 64, 1, input);
        worker.Execute(tensor);
        Tensor output = worker.PeekOutput(); // shape: (1, 1, 1, 8)

        float[] scores = output.ToReadOnlyArray();

        // Optional: view raw output
        if(debug) Debug.Log("Raw output: " + string.Join(", ", scores.Select(s => s.ToString("F4"))));

        // 5. Apply softmax
        float[] probabilities = Softmax(scores);

        // 6. Choose top emotion
        int topIndex = probabilities.ToList().IndexOf(probabilities.Max());
        string emotion = emotionLabels[topIndex];

        Debug.Log($"Predicted Emotion: {emotion}");

        if (emotionTextUI != null)
        {
            emotionTextUI.text = $"Emotion: {emotion}";
        }

        tensor.Dispose();
        output.Dispose();
    }

    float[] Softmax(float[] logits)
    {
        float maxLogit = logits.Max();
        float[] exps = logits.Select(v => Mathf.Exp(v - maxLogit)).ToArray();
        float sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }

    Texture2D CropCenterSquare(Texture2D tex)
    {
        int size = Mathf.Min(tex.width, tex.height);
        int x = (tex.width - size) / 2;
        int y = (tex.height - size) / 2;

        Color[] pixels = tex.GetPixels(x, y, size, size);
        Texture2D cropped = new Texture2D(size, size);
        cropped.SetPixels(pixels);
        cropped.Apply();
        return cropped;
    }

    Texture2D ResizeTo64x64(Texture2D source)
    {
        RenderTexture rt = RenderTexture.GetTemporary(64, 64);
        Graphics.Blit(source, rt);
        RenderTexture.active = rt;
        Texture2D result = new Texture2D(64, 64);
        result.ReadPixels(new Rect(0, 0, 64, 64), 0, 0);
        result.Apply();
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
