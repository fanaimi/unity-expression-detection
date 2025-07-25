using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WebcamDisplayer : MonoBehaviour
{
    public RawImage rawImage;         // for UI display
    public Renderer meshRenderer;     // for 3D object display
    private WebCamTexture webcamTexture;

    void Start()
    {
        // Get default webcam
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            // webcamTexture = new WebCamTexture();
            webcamTexture = new WebCamTexture(640, 360, 30);
            // Optionally choose a specific device
            // webcamTexture = new WebCamTexture(devices[0].name);

            webcamTexture.Play();

            rawImage.texture = webcamTexture;
            rawImage.material = null; // ✅ ensure no interference

            // Mirror horizontally
            rawImage.rectTransform.localScale = new Vector3(-1, 1, 1);

            // Optional: Flip Y if needed
            if (webcamTexture.videoVerticallyMirrored)
            {
                rawImage.rectTransform.localScale = new Vector3(-1, -1, 1);
            }

            // Optional: Fix rotation
            rawImage.rectTransform.localEulerAngles = new Vector3(0, 0, -webcamTexture.videoRotationAngle);

            if (meshRenderer != null)
            {
                meshRenderer.material.mainTexture = webcamTexture;
            }
        }
        else
        {
            Debug.LogWarning("No webcam detected!");
        }
    }

    void OnDisable()
    {
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }
    }
}
