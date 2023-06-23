using System;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

namespace UserInTheBox
{
    [Serializable]
    public class TimeOptions
    {
        public int sampleFrequency;
        public int timeScale;
        public float timestep;
        public float fixedDeltaTime;
    }
    
    [Serializable]
    public class SimulatedUserState
    {
        public bool reset;
        public bool isFinished;
        public bool quitApplication;
        public Vector3 headsetPosition;
        public Quaternion headsetRotation;
        public Vector3 leftControllerPosition;
        public Quaternion leftControllerRotation;
        public Vector3 rightControllerPosition;
        public Quaternion rightControllerRotation;
        public float currentTimestep;
        public float nextTimestep;
    }

    [Serializable]
    public class Observation
    {
        public bool isFinished;
        public float reward;
        public byte[] image;
        public float timeFeature;
    }

    public class ZmqServer
    {
        private ResponseSocket _socket;
        private SimulatedUserState _simulationState;
        private Observation _gameObservation;
        private TimeOptions _timeOptions;
        private int _timeOutSeconds;
        
        public ZmqServer(string port, int timeOutSeconds)
        {
            Debug.Log("Starting up ZMQ server at tcp://localhost:" + port);
            _socket = new ResponseSocket("@tcp://localhost:" + port);
            _gameObservation = new Observation();
            _timeOutSeconds = timeOutSeconds;
        }

        public void Close()
        {
            if (_socket != null)
            {
                _socket.Close();
                NetMQConfig.Cleanup(false);
            }
        }
        
        ~ZmqServer()
        {
            Close();
        }

        public SimulatedUserState ReceiveState()
        {
            // Receive the message and save it into _simulationState
            Receive(out _simulationState);
            
            // Return the parsed state
            return _simulationState;
        }
        
        private void Receive<TMessage>(out TMessage state)
        {
            // Receive message as string  
            string strMessage;
            var success = _socket.TryReceiveFrameString(TimeSpan.FromSeconds(_timeOutSeconds), out strMessage);
            
            // If we time out, shut down
            if (!success)
            {
                Debug.Log("Server timed out, no message received");
                Application.Quit();
            }
            
            // Parse string into an object and save into 'state'
            state = JsonUtility.FromJson<TMessage>(strMessage);
        }

        public void SendObservation(bool isFinished, float reward, byte[] image, float timeFeature)
        {
            // Populate reply
            _gameObservation.isFinished = isFinished;
            _gameObservation.reward = reward;
            _gameObservation.image = image;
            _gameObservation.timeFeature = timeFeature;

            // Send to simulator
            _socket.SendFrame(JsonUtility.ToJson(_gameObservation));
        }
        
        public TimeOptions WaitForHandshake()
        {
            Debug.Log("Waiting for User-in-the-Box to confirm connection");
            // Receive time options from User-in-the-Box
            Receive(out _timeOptions);
            Debug.Log("Connection confirmed");
            
            // Send an empty message to confirm connection
            SendObservation(false, 0, null, -1);

            return _timeOptions;
        }

        public SimulatedUserState GetSimulationState()
        {
            return _simulationState;
        }
    }
}