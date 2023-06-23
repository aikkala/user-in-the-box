using System;
using UnityEngine;


namespace UserInTheBox
{
    public class RLEnv_Whacamole : RLEnv
    {
        // This class implements the RL environment for the Whacamole game.

        public SequenceManager sequenceManager;
        private float _previousPoints, _initialPoints, _elapsedTimeScaled;
        private Transform _marker;
        private string _condition;
        private int _fixedSeed;

        public override void InitialiseReward()
        {
            _initialPoints = sequenceManager.Points;
            _previousPoints = _initialPoints;
            _marker = simulatedUser.rightHandController.Find("Hammer/marker").transform;
            
        }

        public override void InitialiseGame()
        {
            
            // Get game variant and level
            if (!simulatedUser.isDebug())
            {
                _condition = UitBUtils.GetKeywordArgument("condition");
                _logging = UitBUtils.GetOptionalArgument("logging");

                string fixedSeed = UitBUtils.GetOptionalKeywordArgument("fixedSeed", "0");
                // Try to parse given fixed seed string to int
                if (!Int32.TryParse(fixedSeed, out _fixedSeed))
                {
                    Debug.Log("Couldn't parse fixed seed from given value, using default 0");
                    _fixedSeed = 0;
                }

            }
            else
            {
                _condition = "medium";
                _fixedSeed = 0;
                _logging = false;
            }
            Debug.Log("RLEnv set to condition " + _condition);

        }

        public override void UpdateIsFinished()
        {
            // Update finished
            _isFinished = sequenceManager.stateMachine.currentState.Equals(GameState.Ready);
        }

        protected override void CalculateReward()
        {
            // Get hit points
            int points = sequenceManager.Points;
            _reward = (points - _previousPoints)*10;
            _previousPoints = points;
            
            // Also calculate distance component
            foreach (var target in sequenceManager.targetArea.GetComponentsInChildren<Target>())
            {
                if (target.stateMachine.currentState == TargetState.Alive)
                {
                    var dist = Vector3.Distance(target.transform.position, _marker.position);
                    _reward += (float)(Math.Exp(-10*dist)-1) / 10;
                }
            }
        }

        public override float GetTimeFeature()
        {
            return sequenceManager.GetTimeFeature();
        }


        public override void Reset()
        {
            // Set play level
            sequenceManager.playParameters.condition = _condition;
            sequenceManager.playParameters.Initialise(_fixedSeed);
            
            // Visit Ready state, as some important stuff will be set (on exit)
            sequenceManager.stateMachine.GotoState(GameState.Ready);

            // Start playing
            sequenceManager.stateMachine.GotoState(GameState.Play);
            
            // Reset points
            _previousPoints = sequenceManager.Points;
        }
    }
}