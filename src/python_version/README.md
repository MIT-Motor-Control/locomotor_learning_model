# Python Implementation of Locomotor Learning Model

This folder contains a Python implementation of the locomotor learning model that was originally developed in MATLAB. The model simulates how humans adapt their walking patterns on a treadmill, particularly focusing on split-belt treadmill protocols where each leg experiences different speeds.

## Overview

The model represents a biomechanical walking system that:
1. Simulates biped walking dynamics using an inverted pendulum model
2. Incorporates motor learning and adaptation
3. Models energy optimization during gait
4. Adapts to changing walking conditions (e.g., split-belt vs. tied-belt)

## Running the Simulation

To run the simulation:

```bash
python run_simulation.py
```

This will execute the entire simulation pipeline and generate plots showing the adaptation process.

## Main Components

- `rootSimulateLearningWhileWalking.py`: Main entry point that initializes parameters and runs the simulation
- `simulateLearningStepByStep.py`: Core learning algorithm implementation
- `simulateManySteps_AsymmetricControl.py`: Simulates multiple walking steps with different control for each leg
- `fObjective_AsymmetricNominal.py`: Computes the objective function (energy, symmetry, etc.)
- `postProcessAfterLearning.py`: Post-processes simulation results and generates plots

## Parameter Files

- `loadBipedModelParameters.py`: Biomechanical parameters of the biped model
- `loadControllerGainParameters.py`: Controller feedback gains
- `loadLearnerParameters.py`: Learning algorithm parameters
- `loadProtocolParameters.py`: Experimental protocol parameters (treadmill speeds, etc.)
- `loadStoredMemoryParameters_ControlVsSpeed.py`: Initial memory parameters

## Dependencies

- NumPy
- SciPy
- Matplotlib

## Notes

This Python implementation aims to replicate the functionality of the original MATLAB code. Some of the more complex numerical aspects (like the recursive least squares implementation) use simplified approximations that may need further refinement for exact replication of the MATLAB results.
