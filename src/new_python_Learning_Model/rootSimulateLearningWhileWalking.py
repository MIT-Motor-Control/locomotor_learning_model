import matplotlib.pyplot as plt

# Import the functions from other scripts
from load_biped_model_parameters import load_biped_model_parameters
from load_sensory_noise_parameters import load_sensory_noise_parameters
from load_controller_gain_parameters import load_controller_gain_parameters
from load_learner_parameters import load_learner_parameters
from load_protocol_parameters import load_protocol_parameters

# Main script
if __name__ == "__main__":
    # Initialize empty parameter values
    param_fixed = {}

    # Initialize the random number generator
    import random
    random.seed()

    # Load biped model parameters
    param_fixed = load_biped_model_parameters(param_fixed)

    # Load sensory noise parameters
    param_fixed = load_sensory_noise_parameters(param_fixed)

    # Load controller gain parameters
    param_controller_gains = load_controller_gain_parameters(param_fixed)

    # Load learner parameters
    param_fixed = load_learner_parameters(param_fixed)

    # Load protocol parameters
    param_fixed = load_protocol_parameters(param_fixed)

    # Display the parameters for verification
    print("Biped Model Parameters:", param_fixed)
    print("Controller Gain Parameters:", param_controller_gains)

    # You can add further simulation logic here if needed
    # For example, simulate walking, update learning, etc.

    print("Simulation completed successfully.")
