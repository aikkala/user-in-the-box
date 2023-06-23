using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace UserInTheBox
{
    public class Logger : MonoBehaviour
    {
        private Dictionary<string, StreamWriter> _files;
        private string _baseLogFolder;
        private string _subjectFolder;
        private string _experimentFolder;

        public void Awake()
        {
            string outputFolder = UitBUtils.GetOptionalKeywordArgument("outputFolder", 
                Application.persistentDataPath);
            
            _baseLogFolder = Path.Combine(outputFolder, "logging/" + DateTime.Now.ToString("yyyy-MM-dd"));
            Debug.Log("Logs will be saved to " + _baseLogFolder);
            
            // Initialise stream holder
            _files = new Dictionary<string, StreamWriter>();
        }

        public string GenerateSubjectFolder()
        {
            // Generate a new subject folder based on timestamp
            string subjectId = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
            
            // Create a folder for this subject
            _subjectFolder = Path.Combine(_baseLogFolder, subjectId);
            Directory.CreateDirectory(_subjectFolder);

            return subjectId;
        }

        public void GenerateExperimentFolder(string experimentName)
        {
            // Make sure subject folder exists
            if (_subjectFolder == null)
            {
                GenerateSubjectFolder();
            }
            
            if (_experimentFolder == null)
            {
                // Generate a new experiment folder based on timestamp
                string time = DateTime.Now.ToString("HH-mm-ss");
                
                // Create a folder for this experiment
                _experimentFolder = Path.Combine(_subjectFolder, time + "-" + experimentName);
                Directory.CreateDirectory(_experimentFolder);
            }
        }

        public void Initialise(string key)
        {
            // If subject folder is null, generate a new subject
            if (_subjectFolder == null)
            {
                GenerateSubjectFolder();
            }
            
            // If experiment folder is null, generate a new experiment
            if (_experimentFolder == null)
            {
                GenerateExperimentFolder("undefined");
            }
            
            if (!_files.ContainsKey(key))
            {
                string logPath = Path.Combine(_experimentFolder, key + ".csv");
                _files.Add(key, new StreamWriter(logPath));
            }
            else
            {
                throw new IOException("A log file corresponding to key " + key + " has already been initialised");
            }
        }

        public void Finalise(string key)
        {
            if (_files.ContainsKey(key))
            {
                _files[key].Close();
                _files.Remove(key);
            }
        }

        // public async void Push(string key, string msg)
        public void Push(string key, string msg)
        {
            // Do we want async here? Does this even work in Unity?
            if (_files.ContainsKey(key))
            {
                // await _files[key].WriteLineAsync(msg);
                _files[key].WriteLine(msg);
            }
        }

        public void PushWithTimestamp(string key, string msg)
        {
            Push(key, Time.time + ", " + msg);
        }
    }
}
