using System;
using UnityEngine;

namespace UserInTheBox
{

    public abstract class RLEnv : MonoBehaviour
    {
        // This class shows which methods need to be separately implemented for each different game, as we don't know the game
        // dynamics (how/where rewards are received from, which state game is in, etc.).
        // Create a child class for each game and implement the abstract classes to match the game dynamics.

        public SimulatedUser simulatedUser;
        public Logger logger;

        protected float _reward;
        protected bool _isFinished;

        protected bool _logging;

        public void Start()
        {
            // Don't run RLEnv if it is not needed
            if (!simulatedUser.enabled)
            {
                gameObject.SetActive(false);
                return;
            }

            _reward = 0.0f;

            InitialiseGame();
            InitialiseReward();

            // Enable logging if necessary
            logger.enabled = _logging;
        }

        public void Update()
        {
            // Update reward
            CalculateReward();
            
            // Update finished
            UpdateIsFinished();
        }
        
        public float GetReward()
        {
            return _reward;
        }

        public bool IsFinished()
        {
            return _isFinished;
        }

        public abstract void InitialiseGame();
        public abstract void InitialiseReward();
        protected abstract void CalculateReward();
        public abstract void UpdateIsFinished();
        public abstract float GetTimeFeature();
        public abstract void Reset();

    }
}