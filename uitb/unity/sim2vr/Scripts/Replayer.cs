using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem.XR;
using UserInTheBox;
using Logger = UserInTheBox.Logger;

public class Replayer : MonoBehaviour
{
    private class StateData
    { 
        public float timestamp { get; set; }
        public Vector3 leftControllerPosition { get; set; }
        public Vector3 rightControllerPosition { get; set; }
        public Vector3 headsetPosition { get; set; }
        public Quaternion leftControllerRotation { get; set; }
        public Quaternion rightControllerRotation { get; set; }
        public Quaternion headsetRotation { get; set; }
    }

    private class EventData
    {
        public float timestamp { get; set; }
        public string eventType { get; set; }
        public int targetId { get; set; }
        public int gridId { get; set; }
    }
    
    public SequenceManager sequenceManager;
    public SimulatedUser simulatedUser;
    public Logger logger;
    private List<StateData> _stateData;
    private List<EventData> _eventData;
    private StreamReader _reader;
    private float _startTime;
    private int _stateIdx = 1;  // Starts from 1 because index 0 contains initial position
    private int _eventIdx = 0;
    private int _playbackRate;
    private bool _debug = false;

    private void Awake()
    {
        // Check if replayer is enabled
        if (_debug)
        {
            enabled = true;
        }
        else
        {
            enabled = UitBUtils.GetOptionalArgument("replay");
        }
        
        if (enabled)
        {
            // Disable logger when replaying
            logger.enabled = false;

            // Get playback rate (sampling rate)
            string playbackRate = UitBUtils.GetOptionalKeywordArgument("playbackRate", "20");
            
            // Try to parse given sampling rate string to int
            if (!Int32.TryParse(playbackRate, out _playbackRate))
            {
                Debug.Log("Couldn't parse playback rate from given value, using default 20");
                _playbackRate = 20;
            }
        }

    }

    void Start()
    {
        // Set max delta time
        // Time.fixedDeltaTime = 0.01f;
        Time.maximumDeltaTime = 1f / _playbackRate;
        //Application.targetFrameRate = _playbackRate;

        // Disable TrackedPoseDriver, otherwise XR Origin will always try to reset position of camera to (0,0,0)?
        simulatedUser.mainCamera.GetComponent<TrackedPoseDriver>().enabled = false;

        // Get state file path
        string stateLogFilepath = UitBUtils.GetKeywordArgument("stateLogFilepath");
        // string stateLogFilepath = "/home/aleksi/Desktop/testdata/2023-04-17/1681747528/19-05-28-medium/states.csv";
        
        // Parse state log file
        string info = ParseStateLogFile(stateLogFilepath);
        
        // Set initial controller/headset positions/rotations
        UpdateAnchors(_stateData[0]);
        
        // Get start time from first actual datum
        _startTime = _stateData[1].timestamp;
        
        // Initialise play
        InitialisePlay(info);
        
        // Get event file path
        string eventLogFilepath = UitBUtils.GetKeywordArgument("eventLogFilepath");
        // string eventLogFilepath = "/home/aleksi/Desktop/testdata/2023-04-17/1681747528/19-05-28-medium/events.csv";

        // Parse event log file
        ParseEventLogFile(eventLogFilepath);
    }

    private string ParseStateLogFile(string filepath)
    {
        // Read log file (csv) 
        var reader = new StreamReader(filepath);
        
        // First line contains level info
        string info = reader.ReadLine();
        
        // Second line contains the header
        string header = reader.ReadLine();
        
        // Check header matches what we expect
        if (header != sequenceManager.GetStateHeader())
        {
            throw new InvalidDataException("Header of log file " + filepath +
                                           " does not match the header set in SequenceManager.GetStateHeader()");
        }

        // Read rest of file
        _stateData = new List<StateData>();
        while(!reader.EndOfStream)
        {
            // Read line
            string[] values = reader.ReadLine().Split(",");
            
            // Parse and add data to list
            _stateData.Add(new StateData
            {
                timestamp = Str2Float(values[0]),
                leftControllerPosition = Str2Vec3(values[1], values[2], values[3]),
                leftControllerRotation = Str2Quat(values[4], values[5], values[6], values[7]),
                rightControllerPosition = Str2Vec3(values[8], values[9], values[10]),
                rightControllerRotation = Str2Quat(values[11], values[12], values[13], values[14]),
                headsetPosition = Str2Vec3(values[15],values[16], values[17]),
                headsetRotation = Str2Quat(values[18], values[19],values[20], values[21])
            });
        }

        // Return level info
        return info;
    }

    private void ParseEventLogFile(string filepath)
    {
        // Read log file (csv) 
        var reader = new StreamReader(filepath);
        
        // There are no headers, start parsing the file
        _eventData = new List<EventData>();
        while(!reader.EndOfStream)
        {
            // Read line
            string[] values = reader.ReadLine().Split(", ");
            
            // Parse and add data to list -- only target hits are needed
            if (values[1] == "target_hit")
            {
                string[] targetId = values[2].Split(" ");
                _eventData.Add(new EventData
                {
                    timestamp = Str2Float(values[0]),
                    eventType = "target_hit",
                    targetId = Str2Int(targetId[2])
                });
            } else if (values[1] == "target_spawn")
            {
                string[] gridId = values[3].Split(" ");
                _eventData.Add(new EventData
                {
                    timestamp = Str2Float(values[0]),
                    eventType = "target_spawn",
                    gridId = Str2Int(gridId[2])
                });
            }
        }
    }
    
    public float Str2Float(string value)
    {
        return float.Parse(value);
    }

    public int Str2Int(string value)
    {
        return int.Parse(value);
    }

    public Vector3 Str2Vec3(string x, string y, string z)
    {
        return new Vector3(Str2Float(x), Str2Float(y), Str2Float(z));
    }

    public Quaternion Str2Quat(string x, string y, string z, string w)
    {
        return new Quaternion(Str2Float(x), Str2Float(y), Str2Float(z), Str2Float(w));
    }

    private void InitialisePlay(string info)
    {
        // Do some parsing
        string[] values = info.Split(", ");
        string condition = values[0].Split(" ")[1];
        int randomSeed = int.Parse(values[1].Split(" ")[2]);
        
        // Initialise state
        sequenceManager.SetCondition(condition, randomSeed);
    }

    // Update is called once per frame
    void Update()
    {
        // Stop application once all data has been played, and state changed to Ready
        if (_stateIdx >= _stateData.Count && sequenceManager.stateMachine.currentState == GameState.Ready)
        {
            //If we are running in a standalone build of the game
#if UNITY_STANDALONE
             Application.Quit();
#endif
 
            //If we are running in the editor
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
        
        // Do nothing if we ran out of log data (wait for game to close), or wait until Time.time catches up
        if (_stateIdx >= _stateData.Count || _stateData[_stateIdx].timestamp - _startTime > Time.time)
        {
            return;
        }
        
        // Find next timestamp in state data
        while (_stateIdx < _stateData.Count && _stateData[_stateIdx].timestamp - _startTime < Time.time)
        {
            _stateIdx += 1;
        }
        
        // Update anchors
        if (_stateIdx < _stateData.Count)
        {
            UpdateAnchors(_stateData[_stateIdx]);
        }
        
        // Check if we should hit a target (needed as backup, due to how timing works while replaying some hits may go
        // unnoticed -- hit velocity depends on timing of the replayed data, which is only an approximation) or spawn
        // a new target (relying on random seed works when only 1 target at a time is alive, but seems to fail sooner
        // or later when max number of targets > 1)
        while (_eventIdx < _eventData.Count && _stateData[_stateIdx].timestamp >= _eventData[_eventIdx].timestamp)
        {
            if (_eventData[_eventIdx].eventType == "target_hit")
            {
                if (sequenceManager.targetArea.objects.ContainsKey(_eventData[_eventIdx].targetId))
                {
                    // Hit the target
                    sequenceManager.targetArea.objects[_eventData[_eventIdx].targetId].Hit();
                }
            } else if (_eventData[_eventIdx].eventType == "target_spawn")
            {
                sequenceManager.targetArea.SpawnTarget(_eventData[_eventIdx].gridId);
            }
            
            // Move to next event
            _eventIdx += 1;
        }
    }

    private void UpdateAnchors(StateData data)
    {
        simulatedUser.mainCamera.transform.SetPositionAndRotation(data.headsetPosition, data.headsetRotation);
        simulatedUser.leftHandController.SetPositionAndRotation(data.leftControllerPosition, 
            data.leftControllerRotation);
        simulatedUser.rightHandController.SetPositionAndRotation(data.rightControllerPosition, 
            data.rightControllerRotation);
        
        // Camera is looking a bit too high when replayed for some reason => rotate slightly downwards. We can do this
        // because the env/game does not use camera rotation for anything
        simulatedUser.mainCamera.transform.Rotate(new Vector3(10, 0, 0));
    }
}
