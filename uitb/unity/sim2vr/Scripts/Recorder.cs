using System;
using System.IO;
using UnityEngine;
using UserInTheBox;

public class Recorder : MonoBehaviour
{
    public SimulatedUser simulatedUser;
    private Camera _envCamera;
    private Camera _mainCamera;
    private RenderTexture _lightMap;
    private RenderTexture _renderTexture;
    private Texture2D _tex;
    private Rect _rect;
    private int _width;
    private int _height;
    private int _index;
    private string _baseImageFolder;
    private string _envCameraFolder;
    private string _mainCameraFolder;
    private string _resolution;
    private bool _debug = false;

    private void Awake()
    {
        if (_debug)
        {
            enabled = true;
            _baseImageFolder = Path.Combine(Application.persistentDataPath, "recording/");
            _resolution = "1280x960";
        }
        else
        {
            enabled = UitBUtils.GetOptionalArgument("record");
        }

        if (enabled)
        {
            _resolution = "1280x960";
            if (enabled && !_debug)
            {
                _baseImageFolder = Path.Combine(UitBUtils.GetKeywordArgument("outputFolder"), "recording/");
                _resolution = UitBUtils.GetOptionalKeywordArgument("resolution", _resolution);
            }

            Debug.Log("Game play recording is enabled");

            Debug.Log("Images of game play will be saved to " + _baseImageFolder);
            Debug.Log("Game play recording resolution is " + _resolution);

            // Create separate folder for env camera and headset camera
            _envCameraFolder = Path.Combine(_baseImageFolder, "envCamera/");
            _mainCameraFolder = Path.Combine(_baseImageFolder, "mainCamera/");

            // Try to convert given resolution string to ints
            if (!Int32.TryParse(_resolution.Split("x")[0], out _width) ||
                !Int32.TryParse(_resolution.Split("x")[1], out _height))
            {
                Debug.Log("Couldn't parse resolution from given string, using default 1280x960");
                _width = 1280;
                _height = 960;
            }
        }
        else
        {
            gameObject.SetActive(false);
        }
    }

    void Start()
    {
        // Need this stupid hack to make rendered textures lighter (see answer by Invertex in
        // https://forum.unity.com/threads/writting-to-rendertexture-comes-out-darker.427631/)
        _lightMap = new RenderTexture(_width, _height, 16);
        _lightMap.name = "stupid_hack";
        _lightMap.enableRandomWrite = true;
        _lightMap.Create();

        // Get the env camera
        _envCamera = GetComponent<Camera>();
        
        // Get headset camera
        _mainCamera = simulatedUser.mainCamera;
        
        // Create the actual render texture
        _renderTexture = new RenderTexture(_width, _height, 16, RenderTextureFormat.ARGBHalf);

        // Create 2D texture into which we copy from render texture
        _tex = new Texture2D(_width, _height, TextureFormat.RGB24, false);

        _rect = new Rect(0, 0, _width, _height);

        // Delete existing directory
        if (Directory.Exists(_baseImageFolder))
        {
            Directory.Delete(_baseImageFolder, true);
        }

        // Create the output directory
        Directory.CreateDirectory(_baseImageFolder);

        // Also create separate directories for env camera and headset camera
        Directory.CreateDirectory(_envCameraFolder);
        Directory.CreateDirectory(_mainCameraFolder);

        // A running index for image names
        _index = 0;
    }

    void LateUpdate()
    {
        // Capture first image from the env camera
        CaptureEnvCameraImage();

        // Then capture image from headset camera
        CaptureMainCameraImage();

        _index += 1;
    }

    private void CaptureEnvCameraImage()
    {
        // Manually render the env camera
        _envCamera.targetTexture = _renderTexture;
        _envCamera.Render();
        
        // Read pixels first into _lightMap
        RenderTexture.active = _lightMap;
        
        // Blit through _lightMap to make the rendered image lighter
        GL.Clear(true, true, Color.black);
        Graphics.Blit(_renderTexture, _lightMap);

        // Read pixels into tex
        _tex.ReadPixels(_rect, 0, 0);

        // Reset active render texture
        RenderTexture.active = null;

        // Encode texture into PNG
        var image = _tex.EncodeToPNG();
        File.WriteAllBytes(_envCameraFolder + "image" + _index + ".png", image);
    }

    private void CaptureMainCameraImage()
    {
        // Manually render the main camera
        _mainCamera.targetTexture = _renderTexture;
        _mainCamera.Render();
        
        // Read pixels first into _lightMap
        RenderTexture.active = _lightMap;
        
        // Blit through _lightMap to make the rendered image lighter
        GL.Clear(true, true, Color.black);
        Graphics.Blit(_renderTexture, _lightMap);

        // Read pixels into tex
        _tex.ReadPixels(_rect, 0, 0);

        // Reset active render texture
        RenderTexture.active = null;

        // Encode texture into PNG
        var image = _tex.EncodeToPNG();
        File.WriteAllBytes(_mainCameraFolder + "image" + _index + ".png", image);
    }

}
