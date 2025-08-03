using UnityEngine;
using Unity.Barracuda;
using System.IO;

public class FaceDetector : MonoBehaviour
{
    public NNModel modelAsset;
    public RectTransform boundingBoxRect;
    public Texture2D lastInputTexture; // Set externally (WebcamDisplayer)

    private Model model;
    private IWorker worker;

    public void FaceDetectorSetUp()
    {
        Debug.Log("[FaceDetector] Setting up model...");
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
        Debug.Log($"[FaceDetector] Worker created. Model input: {model.inputs[0].name}");
    }

    public void RunFaceDetection(Tensor input)
    {
        if (worker == null)
        {
            Debug.LogError("[FaceDetector] Worker is not initialized!");
            return;
        }

        Debug.Log("[FaceDetector] Running inference...");
        worker.Execute(input);

        using (Tensor scores = worker.PeekOutput("scores"))
        using (Tensor boxes = worker.PeekOutput("boxes"))
        {
            Debug.Log($"[FaceDetector] Output shapes — Scores: {scores.shape}, Boxes: {boxes.shape}");

            int anchors = scores.shape[2];
            float bestScore = 0f;
            int bestIndex = -1;

            for (int i = 0; i < anchors; ++i)
            {
                float faceScore = scores[0, 0, i, 1];
                if (faceScore > bestScore)
                {
                    bestScore = faceScore;
                    bestIndex = i;
                }
            }

            if (bestScore > 0.05f && bestIndex != -1)
            {
                float x1 = boxes[0, 0, bestIndex, 0]; // normalized
                float y1 = boxes[0, 0, bestIndex, 1];
                float x2 = boxes[0, 0, bestIndex, 2];
                float y2 = boxes[0, 0, bestIndex, 3];

                float width = x2 - x1;
                float height = y2 - y1;

                Debug.Log($"[FaceDetector] Face detected. Score={bestScore:F2}, Box=({x1:F2}, {y1:F2}) - ({x2:F2}, {y2:F2})");

                // Map to UI coordinates
                float rectX = (x1 + width / 2f - 0.5f) * 640f;
                float rectY = (0.5f - (y1 + height / 2f)) * 480f;
                float rectW = width * 640f;
                float rectH = height * 480f;

                // 🔳 Draw bounding box in UI
                if (boundingBoxRect != null)
                {
                    boundingBoxRect.anchoredPosition = new Vector2(rectX, rectY);
                    boundingBoxRect.sizeDelta = new Vector2(rectW, rectH);
                    boundingBoxRect.gameObject.SetActive(true);
                    Debug.Log($"[FaceDetector] Bounding box shown at {rectX},{rectY} size {rectW}x{rectH}");
                }

                // 🎯 Save cropped face image
                if (lastInputTexture != null)
                {
                    int imgW = lastInputTexture.width;
                    int imgH = lastInputTexture.height;

                    int px = Mathf.Clamp(Mathf.RoundToInt(x1 * imgW), 0, imgW - 1);
                    int py = Mathf.Clamp(Mathf.RoundToInt((1f - y2) * imgH), 0, imgH - 1);
                    int pw = Mathf.Clamp(Mathf.RoundToInt(width * imgW), 1, imgW - px);
                    int ph = Mathf.Clamp(Mathf.RoundToInt(height * imgH), 1, imgH - py);

                    Debug.Log($"[FaceDetector] Cropping face at px={px}, py={py}, pw={pw}, ph={ph}");

                    try
                    {
                        Color[] facePixels = lastInputTexture.GetPixels(px, py, pw, ph);
                        Texture2D faceImage = new Texture2D(pw, ph, TextureFormat.RGB24, false);
                        faceImage.SetPixels(facePixels);
                        faceImage.Apply();

                        string path = Application.dataPath + "/detectedFace.png";
                        File.WriteAllBytes(path, faceImage.EncodeToPNG());
                        Debug.Log("[FaceDetector] Saved cropped face image to " + path);
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError("[FaceDetector] Failed to crop face image: " + ex.Message);
                    }
                }
                else
                {
                    Debug.LogWarning("[FaceDetector] lastInputTexture is null. Cannot save face crop.");
                }
            }
            else
            {
                Debug.Log("[FaceDetector] No faces detected.");
                if (boundingBoxRect != null)
                    boundingBoxRect.gameObject.SetActive(false);
            }
        }
    }

    private void OnDestroy()
    {
        if (worker != null)
        {
            worker.Dispose();
            Debug.Log("[FaceDetector] Worker disposed.");
        }
    }
}
