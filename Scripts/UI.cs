using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public static UIManager Instance;

    [Header("UI")]
    public GameObject panel;
    public GameObject panel2;
    public Image displayImage;
    public RawImage resultImage;

    [Tooltip(
        "Container that defines the maximum visible area of the result. " +
        "If empty, the parent of Result Image will be used."
    )]
    public RectTransform resultImageContainer;

    [Header("Selectable Images")]
    public Sprite[] images;

    [Header("Roboflow")]
    [Tooltip(
        "Development only. Do not include a private API key in a released build."
    )]
    public string apiKey = "YOUR_ROBOFLOW_API_KEY";

    [TextArea(3, 8)]
    public string classes =
        "worker, container, rebar, steel stud, steel plate, beam, column, " +
        "barrier, barricade, helmet, crane, machinery, traffic sign, fence, " +
        "portable toilet, concrete, bolt, bolt hole";

    [Header("Input Image Settings")]
    [Tooltip(
        "Maximum width or height sent to Roboflow. " +
        "The original aspect ratio is preserved."
    )]
    [Min(64)]
    public int maxInputImageSize = 4096;

    [Range(1, 100)]
    public int jpegQuality = 80;

    [Header("Result Image Settings")]
    [Tooltip(
        "Maximum resolution of the downloaded result texture. " +
        "This controls GPU memory, not the visible UI size."
    )]
    [Min(64)]
    public int maxResultTextureSize = 1024;

    [Tooltip(
        "Percentage of the result container that the image may occupy."
    )]
    [Range(0.1f, 1f)]
    public float resultContainerPadding = 0.95f;

    [Header("Request Settings")]
    [Min(30)]
    public int timeoutSeconds = 180;

    private const string Endpoint =
        "https://serverless.roboflow.com/" +
        "mohsen-khosravi-polito-it/workflows/inspection";

    private Sprite currentSprite;
    private Texture2D currentResultTexture;
    private Coroutine requestCoroutine;
    private bool requestInProgress;

    private void Awake()
    {
        Instance = this;

        if (panel != null)
            panel.SetActive(false);

        if (panel2 != null)
            panel2.SetActive(false);
    }

    
    public void ShowPanel(string imageName)
    {
        currentSprite = null;

        if (images == null || images.Length == 0)
        {
            Debug.LogError(
                "No Sprites are assigned to the Images array."
            );
            return;
        }

        foreach (Sprite sprite in images)
        {
            if (sprite != null && sprite.name == imageName)
            {
                currentSprite = sprite;
                break;
            }
        }

        if (currentSprite == null)
        {
            Debug.LogError(
                "Could not find Sprite named: " + imageName
            );
            return;
        }

        if (displayImage != null)
        {
            displayImage.sprite = currentSprite;
            displayImage.preserveAspect = true;
        }

        if (panel != null)
            panel.SetActive(true);
    }

    public void ClosePanel()
    {
        if (panel != null)
            panel.SetActive(false);
    }

    public void CloseResultPanel()
    {
        if (panel2 != null)
            panel2.SetActive(false);
    }

   
    public void OnClickDetect()
    {
        if (requestInProgress)
        {
            Debug.LogWarning(
                "A Roboflow request is already running."
            );
            return;
        }

        if (currentSprite == null)
        {
            Debug.LogError("No Sprite is selected.");
            return;
        }

        string cleanApiKey =
            apiKey == null ? string.Empty : apiKey.Trim();

        if (string.IsNullOrWhiteSpace(cleanApiKey) ||
            cleanApiKey == "YOUR_ROBOFLOW_API_KEY")
        {
            Debug.LogError(
                "Enter a valid Roboflow API key in the Inspector."
            );
            return;
        }

        if (string.IsNullOrWhiteSpace(classes))
        {
            Debug.LogError(
                "The SAM 3 classes value is empty."
            );
            return;
        }

        Texture2D readableTexture;

        try
        {
            readableTexture =
                SpriteToReadableTexture(currentSprite);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "Failed to read the selected Sprite:\n" +
                exception
            );
            return;
        }

        if (readableTexture == null)
        {
            Debug.LogError(
                "Failed to create a readable texture."
            );
            return;
        }

        if (panel2 != null)
            panel2.SetActive(true);

        Debug.Log(
            "Roboflow endpoint: " + Endpoint
        );

        requestInProgress = true;

        requestCoroutine = StartCoroutine(
            SendImage(readableTexture)
        );
    }

    private Texture2D SpriteToReadableTexture(
        Sprite sprite)
    {
        if (sprite == null || sprite.texture == null)
            return null;

        Texture2D sourceTexture = sprite.texture;
        Rect sourceRect = sprite.textureRect;

        int sourceWidth = Mathf.Max(
            1,
            Mathf.RoundToInt(sourceRect.width)
        );

        int sourceHeight = Mathf.Max(
            1,
            Mathf.RoundToInt(sourceRect.height)
        );

        float scaleFactor = 1f;
        int longestSide =
            Mathf.Max(sourceWidth, sourceHeight);

        if (maxInputImageSize > 0 &&
            longestSide > maxInputImageSize)
        {
            scaleFactor =
                (float)maxInputImageSize / longestSide;
        }

        int targetWidth = Mathf.Max(
            1,
            Mathf.RoundToInt(
                sourceWidth * scaleFactor
            )
        );

        int targetHeight = Mathf.Max(
            1,
            Mathf.RoundToInt(
                sourceHeight * scaleFactor
            )
        );

        Debug.Log(
            "Input image: " +
            sourceWidth + "x" + sourceHeight +
            " -> sending " +
            targetWidth + "x" + targetHeight
        );

        RenderTexture temporary =
            RenderTexture.GetTemporary(
                targetWidth,
                targetHeight,
                0,
                RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.sRGB
            );

        temporary.filterMode = FilterMode.Bilinear;

        RenderTexture previous =
            RenderTexture.active;

        try
        {
            Vector2 sourceScale = new Vector2(
                sourceRect.width / sourceTexture.width,
                sourceRect.height / sourceTexture.height
            );

            Vector2 sourceOffset = new Vector2(
                sourceRect.x / sourceTexture.width,
                sourceRect.y / sourceTexture.height
            );

            Graphics.Blit(
                sourceTexture,
                temporary,
                sourceScale,
                sourceOffset
            );

            RenderTexture.active = temporary;

            Texture2D readableTexture =
                new Texture2D(
                    targetWidth,
                    targetHeight,
                    TextureFormat.RGB24,
                    false
                );

            readableTexture.ReadPixels(
                new Rect(
                    0,
                    0,
                    targetWidth,
                    targetHeight
                ),
                0,
                0,
                false
            );

            readableTexture.Apply(false, false);
            readableTexture.filterMode =
                FilterMode.Bilinear;

            return readableTexture;
        }
        finally
        {
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(temporary);
        }
    }

    private IEnumerator SendImage(Texture2D texture)
    {
        byte[] jpegBytes = null;

        try
        {
            jpegBytes =
                texture.EncodeToJPG(jpegQuality);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "Failed to encode the input as JPEG:\n" +
                exception
            );
        }
        finally
        {
            Destroy(texture);
        }

        if (jpegBytes == null ||
            jpegBytes.Length == 0)
        {
            Debug.LogError(
                "JPEG encoding returned no bytes."
            );

            FinishRequest();
            yield break;
        }

        string inputBase64 =
            Convert.ToBase64String(jpegBytes);

        JObject payload = new JObject
        {
            ["inputs"] = new JObject
            {
                ["image"] = new JObject
                {
                    ["type"] = "base64",
                    ["value"] = inputBase64
                },
                ["classes"] = classes
            }
        };

        string requestJson = payload.ToString(
            Newtonsoft.Json.Formatting.None
        );

        byte[] requestBody =
            Encoding.UTF8.GetBytes(requestJson);

        float requestMegabytes =
            requestBody.Length / (1024f * 1024f);

        Debug.Log(
            "Sending image to Roboflow. JPEG bytes: " +
            jpegBytes.Length +
            " | request body: " +
            requestMegabytes.ToString("F2") +
            " MB"
        );

        using (UnityWebRequest request =
               new UnityWebRequest(
                   Endpoint,
                   UnityWebRequest.kHttpVerbPOST
               ))
        {
            request.uploadHandler =
                new UploadHandlerRaw(requestBody);

            request.downloadHandler =
                new DownloadHandlerBuffer();

            request.timeout = timeoutSeconds;

            request.SetRequestHeader(
                "Content-Type",
                "application/json"
            );

            request.SetRequestHeader(
                "Accept",
                "application/json"
            );

            request.SetRequestHeader(
                "Authorization",
                "Bearer " + apiKey.Trim()
            );

            yield return request.SendWebRequest();

            byte[] rawResponse =
                request.downloadHandler != null
                    ? request.downloadHandler.data
                    : null;

            string responseBody =
                request.downloadHandler != null
                    ? request.downloadHandler.text
                    : string.Empty;

            Debug.Log(
                "Roboflow result: " +
                request.result +
                " | HTTP " +
                request.responseCode +
                " | response bytes: " +
                (rawResponse != null
                    ? rawResponse.Length
                    : 0) +
                " | content type: " +
                request.GetResponseHeader(
                    "Content-Type"
                )
            );

            if (request.result ==
                UnityWebRequest.Result.Success)
            {
                HandleResponse(responseBody);
            }
            else
            {
                Debug.LogError(
                    "Roboflow request failed. HTTP " +
                    request.responseCode +
                    ": " +
                    request.error
                );

                if (!string.IsNullOrWhiteSpace(
                        responseBody))
                {
                    Debug.LogError(
                        "Error response:\n" +
                        TruncateForLog(
                            responseBody,
                            4000
                        )
                    );
                }
            }
        }

        FinishRequest();
    }

    private void HandleResponse(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            Debug.LogError(
                "Roboflow returned an empty response."
            );
            return;
        }

        try
        {
            JToken root = JToken.Parse(json);

            JToken firstOutput =
                GetFirstWorkflowOutput(root);

            if (firstOutput == null)
            {
                Debug.LogError(
                    "Could not find a Workflow output.\n" +
                    TruncateForLog(json, 2000)
                );
                return;
            }

            string outputFields =
                GetObjectFieldNames(firstOutput);

            Debug.Log(
                "Roboflow output fields: " +
                outputFields
            );

            JToken imageToken =
                firstOutput["annotated_image"] ??
                firstOutput["vis"];

            if (imageToken == null)
            {
                Debug.LogError(
                    "Roboflow returned no visualization.\n" +
                    "Available output fields: " +
                    outputFields
                );
                return;
            }

            string imageBase64 =
                ExtractBase64Image(imageToken);

            if (string.IsNullOrWhiteSpace(
                    imageBase64))
            {
                Debug.LogError(
                    "The visualization output exists, " +
                    "but no base64 image was found.\n" +
                    "Token type: " +
                    imageToken.Type +
                    "\nToken preview:\n" +
                    TruncateForLog(
                        imageToken.ToString(),
                        2000
                    )
                );
                return;
            }

            Debug.Log(
                "Extracted visualization base64. " +
                "Characters: " +
                imageBase64.Length
            );

            DecodeSaveAndDisplayImage(imageBase64);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "Failed to process the response:\n" +
                exception
            );

            Debug.LogError(
                "Response preview:\n" +
                TruncateForLog(json, 2000)
            );
        }
    }

    private JToken GetFirstWorkflowOutput(
        JToken root)
    {
        if (root == null)
            return null;

        JToken outputs = root["outputs"];

        if (outputs != null)
        {
            if (outputs.Type == JTokenType.Array)
            {
                JArray outputArray =
                    (JArray)outputs;

                if (outputArray.Count > 0)
                {
                    JToken first =
                        outputArray[0];

                    if (first.Type ==
                            JTokenType.Array &&
                        ((JArray)first).Count > 0)
                    {
                        first =
                            ((JArray)first)[0];
                    }

                    return first;
                }
            }

            if (outputs.Type ==
                JTokenType.Object)
            {
                return outputs;
            }
        }

        if (root.Type == JTokenType.Array)
        {
            JArray rootArray = (JArray)root;

            if (rootArray.Count > 0)
            {
                JToken first = rootArray[0];

                if (first.Type ==
                        JTokenType.Array &&
                    ((JArray)first).Count > 0)
                {
                    first = ((JArray)first)[0];
                }

                return first;
            }
        }

        if (root.Type == JTokenType.Object)
        {
            if (root["annotated_image"] != null ||
                root["vis"] != null ||
                root["predictions"] != null)
            {
                return root;
            }
        }

        return null;
    }

    private string ExtractBase64Image(
        JToken token)
    {
        if (token == null ||
            token.Type == JTokenType.Null ||
            token.Type == JTokenType.Undefined)
        {
            return null;
        }

        if (token.Type == JTokenType.String)
        {
            string candidate =
                token.Value<string>();

            return LooksLikeImageData(candidate)
                ? candidate
                : null;
        }

        if (token.Type == JTokenType.Object)
        {
            JObject imageObject =
                (JObject)token;

            string[] preferredKeys =
            {
                "value",
                "base64",
                "image",
                "data",
                "bytes",
                "content"
            };

            foreach (string key in preferredKeys)
            {
                JToken child = imageObject[key];

                if (child == null)
                    continue;

                string result =
                    ExtractBase64Image(child);

                if (!string.IsNullOrWhiteSpace(
                        result))
                {
                    return result;
                }
            }

            foreach (JProperty property
                     in imageObject.Properties())
            {
                if (property.Name == "predictions" ||
                    property.Name == "video_metadata" ||
                    property.Name == "rle_mask" ||
                    property.Name == "counts")
                {
                    continue;
                }

                string result =
                    ExtractBase64Image(
                        property.Value
                    );

                if (!string.IsNullOrWhiteSpace(
                        result))
                {
                    return result;
                }
            }
        }

        if (token.Type == JTokenType.Array)
        {
            foreach (JToken item
                     in token.Children())
            {
                string result =
                    ExtractBase64Image(item);

                if (!string.IsNullOrWhiteSpace(
                        result))
                {
                    return result;
                }
            }
        }

        return null;
    }

    private bool LooksLikeImageData(
        string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return false;

        value = value.Trim();

        if (value.StartsWith(
                "data:image/",
                StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (value.StartsWith("/9j/") ||
            value.StartsWith("iVBOR") ||
            value.StartsWith("UklGR"))
        {
            return true;
        }

        return value.Length > 100 &&
               IsProbablyBase64(value);
    }

    private bool IsProbablyBase64(
        string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return false;

        int commaIndex = value.IndexOf(',');

        if (value.StartsWith(
                "data:",
                StringComparison.OrdinalIgnoreCase) &&
            commaIndex >= 0)
        {
            value = value.Substring(
                commaIndex + 1
            );
        }

        value =
            RemoveBase64Whitespace(value);

        if (value.Length < 4)
            return false;

        foreach (char character in value)
        {
            bool valid =
                character >= 'A' &&
                character <= 'Z' ||

                character >= 'a' &&
                character <= 'z' ||

                character >= '0' &&
                character <= '9' ||

                character == '+' ||
                character == '/' ||
                character == '-' ||
                character == '_' ||
                character == '=';

            if (!valid)
                return false;
        }

        return true;
    }

    private void DecodeSaveAndDisplayImage(
        string base64String)
    {
        if (string.IsNullOrWhiteSpace(
                base64String))
        {
            Debug.LogError(
                "The visualization base64 is empty."
            );
            return;
        }

        try
        {
            base64String = base64String.Trim();

            if (base64String.StartsWith(
                    "data:",
                    StringComparison.OrdinalIgnoreCase))
            {
                int commaIndex =
                    base64String.IndexOf(',');

                if (commaIndex < 0)
                {
                    Debug.LogError(
                        "The image data URL has " +
                        "no comma separator."
                    );
                    return;
                }

                base64String =
                    base64String.Substring(
                        commaIndex + 1
                    );
            }

            base64String =
                RemoveBase64Whitespace(
                    base64String
                );

            base64String = base64String
                .Replace('-', '+')
                .Replace('_', '/');

            int remainder =
                base64String.Length % 4;

            if (remainder == 1)
            {
                Debug.LogError(
                    "The base64 string has an " +
                    "invalid length and may be truncated."
                );
                return;
            }

            if (remainder > 0)
            {
                base64String =
                    base64String.PadRight(
                        base64String.Length +
                        (4 - remainder),
                        '='
                    );
            }

            byte[] imageBytes =
                Convert.FromBase64String(
                    base64String
                );

            if (imageBytes.Length == 0)
            {
                Debug.LogError(
                    "Base64 decoding returned no bytes."
                );
                return;
            }

            Debug.Log(
                "Decoded image bytes: " +
                imageBytes.Length
            );

            Debug.Log(
                "Image byte signature: " +
                GetByteSignature(imageBytes)
            );

            string extension =
                GetImageExtension(imageBytes);

            string outputPath = Path.Combine(
                Application.persistentDataPath,
                "roboflow_output" + extension
            );

            File.WriteAllBytes(
                outputPath,
                imageBytes
            );

            Debug.Log(
                "Saved original Roboflow output to:\n" +
                outputPath
            );

            Texture2D decodedTexture =
                new Texture2D(
                    2,
                    2,
                    TextureFormat.RGBA32,
                    false
                );

            bool loaded =
                decodedTexture.LoadImage(
                    imageBytes,
                    false
                );

            if (!loaded)
            {
                Destroy(decodedTexture);

                Debug.LogError(
                    "Unity could not decode the " +
                    "returned image bytes."
                );
                return;
            }

            Texture2D finalTexture =
                ResizeTexture(
                    decodedTexture,
                    maxResultTextureSize
                );

            if (finalTexture != decodedTexture)
                Destroy(decodedTexture);

            if (currentResultTexture != null)
                Destroy(currentResultTexture);

            currentResultTexture = finalTexture;

            if (resultImage == null)
            {
                Debug.LogError(
                    "Result Image is not assigned."
                );
                return;
            }

            resultImage.texture =
                currentResultTexture;

            resultImage.color = Color.white;
            resultImage.gameObject.SetActive(true);

            /*
             * Disable AspectRatioFitter because this script
             * sizes the RectTransform directly. Using both
             * systems can cause layout conflicts.
             */
            AspectRatioFitter fitter =
                resultImage.GetComponent<
                    AspectRatioFitter
                >();

            if (fitter != null)
                fitter.enabled = false;

            FitResultImageToContainer(
                currentResultTexture
            );

            Debug.Log(
                "Displayed Roboflow result. " +
                "Texture resolution: " +
                currentResultTexture.width +
                "x" +
                currentResultTexture.height
            );
        }
        catch (FormatException exception)
        {
            Debug.LogError(
                "The visualization is not valid base64:\n" +
                exception
            );
        }
        catch (IOException exception)
        {
            Debug.LogError(
                "The image was decoded, but could " +
                "not be saved:\n" +
                exception
            );
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "Failed to decode or display image:\n" +
                exception
            );
        }
    }

    /// <summary>
    /// Downscales the texture resolution while preserving
    /// the aspect ratio. If resizing is unnecessary, the
    /// original texture is returned.
    /// </summary>
    private Texture2D ResizeTexture(
        Texture2D source,
        int maxSize)
    {
        if (source == null)
            return null;

        if (maxSize <= 0)
            return source;

        int originalWidth = source.width;
        int originalHeight = source.height;

        int longestSide = Mathf.Max(
            originalWidth,
            originalHeight
        );

        if (longestSide <= maxSize)
            return source;

        float scale =
            (float)maxSize / longestSide;

        int targetWidth = Mathf.Max(
            1,
            Mathf.RoundToInt(
                originalWidth * scale
            )
        );

        int targetHeight = Mathf.Max(
            1,
            Mathf.RoundToInt(
                originalHeight * scale
            )
        );

        RenderTexture temporary =
            RenderTexture.GetTemporary(
                targetWidth,
                targetHeight,
                0,
                RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.sRGB
            );

        temporary.filterMode =
            FilterMode.Bilinear;

        RenderTexture previous =
            RenderTexture.active;

        try
        {
            Graphics.Blit(source, temporary);
            RenderTexture.active = temporary;

            Texture2D resizedTexture =
                new Texture2D(
                    targetWidth,
                    targetHeight,
                    TextureFormat.RGBA32,
                    false
                );

            resizedTexture.ReadPixels(
                new Rect(
                    0,
                    0,
                    targetWidth,
                    targetHeight
                ),
                0,
                0,
                false
            );

            resizedTexture.Apply(false, false);

            resizedTexture.filterMode =
                FilterMode.Bilinear;

            Debug.Log(
                "Result texture resized: " +
                originalWidth + "x" +
                originalHeight +
                " -> " +
                targetWidth + "x" +
                targetHeight
            );

            return resizedTexture;
        }
        finally
        {
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(temporary);
        }
    }

    /// <summary>
    /// Fits the visible RawImage within its container.
    /// This controls screen size independently of texture resolution.
    /// </summary>
    private void FitResultImageToContainer(
        Texture2D texture)
    {
        if (resultImage == null ||
            texture == null)
        {
            return;
        }

        RectTransform imageRect =
            resultImage.rectTransform;

        RectTransform container =
            resultImageContainer;

        if (container == null)
        {
            container =
                imageRect.parent as RectTransform;
        }

        if (container == null)
        {
            Debug.LogWarning(
                "Result Image has no RectTransform " +
                "container. Assign Result Image Container."
            );
            return;
        }

        Canvas.ForceUpdateCanvases();

        LayoutRebuilder.ForceRebuildLayoutImmediate(
            container
        );

        float padding = Mathf.Clamp(
            resultContainerPadding,
            0.1f,
            1f
        );

        float availableWidth =
            container.rect.width * padding;

        float availableHeight =
            container.rect.height * padding;

        if (availableWidth <= 0f ||
            availableHeight <= 0f)
        {
            Debug.LogError(
                "Result container has invalid dimensions: " +
                container.rect.width +
                "x" +
                container.rect.height
            );
            return;
        }

        float imageAspect =
            (float)texture.width /
            texture.height;

        float containerAspect =
            availableWidth /
            availableHeight;

        float displayWidth;
        float displayHeight;

        if (imageAspect > containerAspect)
        {
            displayWidth = availableWidth;

            displayHeight =
                displayWidth / imageAspect;
        }
        else
        {
            displayHeight = availableHeight;

            displayWidth =
                displayHeight * imageAspect;
        }

        imageRect.anchorMin =
            new Vector2(0.5f, 0.5f);

        imageRect.anchorMax =
            new Vector2(0.5f, 0.5f);

        imageRect.pivot =
            new Vector2(0.5f, 0.5f);

        imageRect.anchoredPosition =
            Vector2.zero;

        imageRect.localScale =
            Vector3.one;

        imageRect.SetSizeWithCurrentAnchors(
            RectTransform.Axis.Horizontal,
            displayWidth
        );

        imageRect.SetSizeWithCurrentAnchors(
            RectTransform.Axis.Vertical,
            displayHeight
        );

        Debug.Log(
            "Result container: " +
            container.rect.width.ToString("F0") +
            "x" +
            container.rect.height.ToString("F0") +
            " | visible image: " +
            displayWidth.ToString("F0") +
            "x" +
            displayHeight.ToString("F0")
        );
    }

    private string RemoveBase64Whitespace(
        string value)
    {
        if (string.IsNullOrEmpty(value))
            return value;

        StringBuilder builder =
            new StringBuilder(value.Length);

        foreach (char character in value)
        {
            if (!char.IsWhiteSpace(character))
                builder.Append(character);
        }

        return builder.ToString();
    }

    private string GetImageExtension(
        byte[] bytes)
    {
        if (bytes == null || bytes.Length < 4)
            return ".bin";

        // JPEG
        if (bytes[0] == 0xFF &&
            bytes[1] == 0xD8 &&
            bytes[2] == 0xFF)
        {
            return ".jpg";
        }

        // PNG
        if (bytes[0] == 0x89 &&
            bytes[1] == 0x50 &&
            bytes[2] == 0x4E &&
            bytes[3] == 0x47)
        {
            return ".png";
        }

        // WEBP
        if (bytes.Length >= 12 &&
            bytes[0] == (byte)'R' &&
            bytes[1] == (byte)'I' &&
            bytes[2] == (byte)'F' &&
            bytes[3] == (byte)'F' &&
            bytes[8] == (byte)'W' &&
            bytes[9] == (byte)'E' &&
            bytes[10] == (byte)'B' &&
            bytes[11] == (byte)'P')
        {
            return ".webp";
        }

        return ".bin";
    }

    private string GetByteSignature(
        byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
            return "(empty)";

        int count = Mathf.Min(
            bytes.Length,
            12
        );

        StringBuilder builder =
            new StringBuilder();

        for (int index = 0;
             index < count;
             index++)
        {
            if (index > 0)
                builder.Append(' ');

            builder.Append(
                bytes[index].ToString("X2")
            );
        }

        return builder.ToString();
    }

    private string GetObjectFieldNames(
        JToken token)
    {
        JObject objectToken =
            token as JObject;

        if (objectToken == null)
            return "(output is not an object)";

        List<string> names =
            new List<string>();

        foreach (JProperty property
                 in objectToken.Properties())
        {
            names.Add(property.Name);
        }

        return names.Count > 0
            ? string.Join(", ", names)
            : "(no fields)";
    }

    private string TruncateForLog(
        string value,
        int maximumLength)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;

        if (value.Length <= maximumLength)
            return value;

        return value.Substring(
                   0,
                   maximumLength
               ) +
               "\n[Response truncated]";
    }

    private void FinishRequest()
    {
        requestInProgress = false;
        requestCoroutine = null;
    }

    private void OnDestroy()
    {
        if (requestCoroutine != null)
        {
            StopCoroutine(requestCoroutine);
            requestCoroutine = null;
        }

        if (currentResultTexture != null)
        {
            Destroy(currentResultTexture);
            currentResultTexture = null;
        }

        if (Instance == this)
            Instance = null;
    }
}
