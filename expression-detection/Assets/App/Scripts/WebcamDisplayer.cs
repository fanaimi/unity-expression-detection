using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using UnityEngine.Experimental.Rendering;

public class WebcamDisplayer : MonoBehaviour
{
    public RawImage rawImage;
    [SerializeField]
    private FaceDetector faceDetector;

    private WebCamTexture webcamTexture;
    public Texture2D CurrentFrameTexture { get; private set; }

    private const int inputWidth = 640;
    private const int inputHeight = 480;

    void Awake()
    {
        Debug.Log("[WebcamDisplayer] Awake called.");
        if (rawImage == null)
        {
            Debug.LogError("[WebcamDisplayer] RawImage not assigned!");
            return;
        }

        webcamTexture = new WebCamTexture(inputWidth, inputHeight);
        rawImage.texture = webcamTexture;
        rawImage.material.mainTexture = webcamTexture;
        Debug.Log("[WebcamDisplayer] RawImage assigned and configured.");
    }

    void Start()
    {
        Debug.Log("[WebcamDisplayer] Initializing webcam...");
        if (webcamTexture != null)
        {
            webcamTexture.Play();
        }

        if (faceDetector != null)
        {
            faceDetector.FaceDetectorSetUp();
        }
        else
        {
            Debug.LogError("[WebcamDisplayer] FaceDetector is not assigned!");
        }

        InvokeRepeating("ProcessFrame", 1f, 0.5f); // process every 0.5s
    }

    public bool IsWebcamReady()
    {
        return webcamTexture != null && webcamTexture.width > 100;
    }

    private void ProcessFrame()
    {
        if (!IsWebcamReady()) return;

        CurrentFrameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
        CurrentFrameTexture.SetPixels32(webcamTexture.GetPixels32());
        CurrentFrameTexture.Apply();

        Debug.Log("[WebcamDisplayer] Captured frame from webcam.");

        using (Tensor inputTensor = PreprocessForUltraFace(CurrentFrameTexture))
        {
            if (faceDetector != null && inputTensor != null)
            {
                faceDetector.RunDetection(inputTensor);
            }
        }
    }

    private Tensor PreprocessForUltraFace(Texture2D tex)
    {
        Texture2D resized = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);
        Color[] pixels = tex.GetPixels();
        Color[] resizedPixels = ResizeBilinear(pixels, tex.width, tex.height, inputWidth, inputHeight);
        resized.SetPixels(resizedPixels);
        resized.Apply();

        Color32[] pixels32 = resized.GetPixels32();
        Tensor tensor = new Tensor(1, inputHeight, inputWidth, 3); // NHWC (Batch, Height, Width, Channels)

        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                int pixelIndex = y * inputWidth + x;
                Color32 pixel = pixels32[pixelIndex];

                tensor[0, y, x, 0] = (pixel.r - 127f) / 128f;
                tensor[0, y, x, 1] = (pixel.g - 127f) / 128f;
                tensor[0, y, x, 2] = (pixel.b - 127f) / 128f;
            }
        }

        Debug.Log("[WebcamDisplayer] Tensor populated as NHWC (1, 480, 640, 3).");
        return tensor;
    }



    private Color[] ResizeBilinear(Color[] original, int w, int h, int newW, int newH)
    {
        Color[] newPixels = new Color[newW * newH];
        float xRatio = (float)w / newW;
        float yRatio = (float)h / newH;

        for (int y = 0; y < newH; y++)
        {
            for (int x = 0; x < newW; x++)
            {
                int px = Mathf.FloorToInt(x * xRatio);
                int py = Mathf.FloorToInt(y * yRatio);
                newPixels[y * newW + x] = original[py * w + px];
            }
        }

        return newPixels;
    }
}
