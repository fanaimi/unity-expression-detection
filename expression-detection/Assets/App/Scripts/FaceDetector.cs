using UnityEngine;
using Unity.Barracuda;

public class FaceDetector : MonoBehaviour
{
    public NNModel modelAsset;
    private Model model;
    private IWorker worker;

    public void FaceDetectorSetUp()
    {
        Debug.Log("[FaceDetector] Setting up model...");
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
        Debug.Log($"[FaceDetector] Worker created. Model input: {model.inputs[0].name}");
    }

    public void RunDetection(Tensor input)
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

            int faceCount = 0;
            for (int i = 0; i < scores.shape[1]; ++i)
            {
                float prob = scores[0, i, 0, 1]; // [N, 4420, 2]
                if (prob > 0.7f)
                {
                    faceCount++;
                    Debug.Log($"[FaceDetector] Face detected with confidence: {prob}");
                }
            }

            if (faceCount == 0)
                Debug.Log("[FaceDetector] No faces detected.");
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
