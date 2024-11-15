import matplotlib.pyplot as plt
import numpy as np 

# Import the functions from other scripts
from load_biped_model_parameters import load_biped_model_parameters
from load_sensory_noise_parameters import load_sensory_noise_parameters
from load_controller_gain_parameters import load_controller_gain_parameters
from load_learner_parameters import load_learner_parameters
from load_protocol_parameters import load_protocol_parameters
from load_stored_memory_parameters_control_vs_speed import load_stored_memory_parameters_control_vs_speed
from load_learnable_parameters_initial import load_learnable_parameters_initial
from load_initial_body_state import load_initial_body_state
from get_treadmill_speed import get_treadmill_speed 
from simulate_learning_initialization import simulate_learning_initialization


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

    # Load initial stored memory, default control
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)

    # Load current learnable parameters
    # Look inside the function to determine which controller parameters are tuned by the learning algorithm
    p_input_controller_asymmetric_nominal = load_learnable_parameters_initial(param_fixed)

    # Load initial state and time
    state_var0_model = load_initial_body_state(p_input_controller_asymmetric_nominal)
    t_start = 0

    # Store initial state for later use
    state_var0_model_before_learning = state_var0_model

    # Context for the gait
    v_a, v_b = get_treadmill_speed(t_start, param_fixed['imposedFootSpeeds'])
    context_now = [v_a, v_b]
    context_length = len(context_now)

    # Simulate learning step by step
    p_input_controller_store_ones_tried = simulate_learning_initialization(
        param_fixed, p_input_controller_asymmetric_nominal,
        state_var0_model, context_now, param_controller_gains
    )

    # Convert the 8D back up to 10D to use the old functions
    p_input_controller_store_8d = p_input_controller_store_ones_tried
    p_input_controller_store_10d = np.vstack((
        p_input_controller_store_8d[:3, :],
        np.zeros((2, p_input_controller_store_8d.shape[1])),
        p_input_controller_store_8d[3:8, :]
    ))



    # Display the parameters for verification
    print("Biped Model Parameters:", param_fixed)
    print("Controller Gain Parameters:", param_controller_gains)

    # You can add further simulation logic here if needed
    # For example, simulate walking, update learning, etc.

    print("Simulation completed successfully.")
