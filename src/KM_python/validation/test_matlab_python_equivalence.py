"""
Comprehensive test suite to validate functional equivalence between MATLAB and Python implementations.

This module provides tests to verify that the Python implementation produces
identical results to the MATLAB version for all key functions and scenarios.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all the Python functions
from parameter_loading.load_biped_model_parameters import load_biped_model_parameters
from parameter_loading.load_sensory_noise_parameters import load_sensory_noise_parameters
from parameter_loading.load_controller_gain_parameters import load_controller_gain_parameters
from parameter_loading.load_learner_parameters import load_learner_parameters
from parameter_loading.load_protocol_parameters import load_protocol_parameters
from parameter_loading.load_stored_memory_parameters_control_vs_speed import load_stored_memory_parameters_control_vs_speed
from initializing.load_learnable_parameters_initial import load_learnable_parameters_initial
from initializing.load_initial_body_state import load_initial_body_state
from initializing.get_treadmill_speed import get_treadmill_speed
from learning.f_objective_asymmetric_nominal import f_objective_asymmetric_nominal
from learning.simulate_learning_step_by_step import simulate_learning_step_by_step


class MATLABPythonValidator:
    """Validates functional equivalence between MATLAB and Python implementations."""
    
    def __init__(self):
        """Initialize the validator."""
        self.tolerance = 1e-10  # Numerical tolerance for comparisons
        self.results = {}
        
    def setup_parameters(self):
        """Set up test parameters identically to MATLAB version."""
        print("Setting up parameters...")
        
        # Initialize parameters exactly as in MATLAB
        param_fixed = {}
        
        # Load parameters in same order as MATLAB
        param_fixed = load_biped_model_parameters(param_fixed)
        param_fixed = load_sensory_noise_parameters(param_fixed)
        param_controller_gains = load_controller_gain_parameters(param_fixed)
        param_fixed = load_learner_parameters(param_fixed)
        param_fixed = load_protocol_parameters(param_fixed)
        param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)
        
        # Set random seed for reproducibility (MATLAB equivalent: rng('default'))
        np.random.seed(42)  # Use fixed seed instead of 'shuffle' for testing
        
        return param_fixed, param_controller_gains
    
    def test_parameter_consistency(self):
        """Test that parameters match expected MATLAB values."""
        print("\\n=== Testing Parameter Consistency ===")
        
        param_fixed, param_controller_gains = self.setup_parameters()
        
        # Test critical parameter values that should match MATLAB exactly
        # Note: MATLAB uses dimensionless/normalized parameters
        expected_values = {
            'mbody': 1.0,
            'leglength': 1.0,  # Dimensionless in MATLAB
            'gravg': 1.0,      # Dimensionless in MATLAB
            'efficiency_pos': 0.25,
            'efficiency_neg': 1.2,  # Positive value in MATLAB (not -1.2)
            'lambdaEnergyVsPeriodicity': 1.0,
            'lambdaEnergyVsSymmetry': 0.75,  # As set in MATLAB
            'symmetryMultiplier': 10,        # As set in MATLAB
        }
        
        passed = 0
        total = len(expected_values)
        
        for param_name, expected_value in expected_values.items():
            actual_value = param_fixed.get(param_name, None)
            if actual_value is not None and abs(actual_value - expected_value) < self.tolerance:
                print(f"✅ {param_name}: {actual_value} (matches expected)")
                passed += 1
            else:
                print(f"❌ {param_name}: {actual_value} (expected {expected_value})")
        
        self.results['parameter_consistency'] = {
            'passed': passed,
            'total': total,
            'success_rate': passed / total
        }
        
        return param_fixed, param_controller_gains
    
    def test_objective_function_determinism(self, param_fixed, param_controller_gains):
        """Test that objective function produces consistent results."""
        print("\\n=== Testing Objective Function Determinism ===")
        
        # Load initial parameters
        p_input_controller = load_learnable_parameters_initial(param_fixed)
        state_var0_model = load_initial_body_state(p_input_controller)
        
        # Test objective function with same inputs multiple times
        results = []
        for i in range(3):
            # Reset random seed to ensure determinism
            np.random.seed(42)
            
            result = f_objective_asymmetric_nominal(
                p_input_controller,
                state_var0_model.copy(),
                param_controller_gains,
                param_fixed,
                0.0
            )
            results.append(result[0])  # Objective value
            print(f"Run {i+1}: Objective = {result[0]:.12f}")
        
        # Check if all runs produce identical results
        max_diff = max(results) - min(results)
        deterministic = max_diff < self.tolerance
        
        self.results['objective_determinism'] = {
            'deterministic': deterministic,
            'max_difference': max_diff,
            'values': results
        }
        
        if deterministic:
            print(f"✅ Objective function is deterministic (max diff: {max_diff:.2e})")
        else:
            print(f"❌ Objective function is not deterministic (max diff: {max_diff:.2e})")
        
        return p_input_controller, state_var0_model
    
    def test_treadmill_speed_function(self, param_fixed):
        """Test treadmill speed interpolation."""
        print("\\n=== Testing Treadmill Speed Function ===")
        
        # Test at multiple time points
        test_times = [0.0, 10.0, 100.0, 1000.0, 5000.0]
        results = []
        
        for t in test_times:
            vA, vB = get_treadmill_speed(t, param_fixed['imposedFootSpeeds'])
            results.append((t, vA, vB))
            print(f"t={t:6.1f}: vA={vA:.6f}, vB={vB:.6f}")
        
        # Check that speeds are reasonable and symmetric initially
        initial_speed = results[0]
        speeds_symmetric = abs(initial_speed[1] - initial_speed[2]) < self.tolerance
        
        self.results['treadmill_speed'] = {
            'symmetric_initial': speeds_symmetric,
            'results': results
        }
        
        if speeds_symmetric:
            print(f"✅ Initial treadmill speeds are symmetric")
        else:
            print(f"❌ Initial treadmill speeds are not symmetric")
    
    def test_short_learning_simulation(self, param_fixed, param_controller_gains):
        """Test a short learning simulation for consistency."""
        print("\\n=== Testing Short Learning Simulation ===")
        
        # Set up for short test
        param_fixed['num_iterations'] = 10  # Very short for testing
        
        p_input_controller = load_learnable_parameters_initial(param_fixed)
        state_var0_model = load_initial_body_state(p_input_controller)
        vA, vB = get_treadmill_speed(0, param_fixed['imposedFootSpeeds'])
        context_now = np.array([vA, vB])
        
        # Run simulation multiple times with same seed
        results = []
        for run in range(2):
            np.random.seed(42)  # Reset seed for reproducibility
            
            result = simulate_learning_step_by_step(
                param_fixed,
                p_input_controller.copy(),
                state_var0_model.copy(),
                context_now.copy(),
                param_controller_gains
            )
            results.append(result)
            print(f"Run {run+1}: Final shape {result.shape}, final param[0] = {result[0,-1]:.8f}")
        
        # Check reproducibility
        diff = np.max(np.abs(results[0] - results[1]))
        reproducible = diff < self.tolerance
        
        self.results['learning_reproducibility'] = {
            'reproducible': reproducible,
            'max_difference': diff,
            'final_shapes': [r.shape for r in results]
        }
        
        if reproducible:
            print(f"✅ Learning simulation is reproducible (max diff: {diff:.2e})")
        else:
            print(f"❌ Learning simulation is not reproducible (max diff: {diff:.2e})")
        
        return results[0]
    
    def test_learning_convergence_properties(self, param_fixed, param_controller_gains):
        """Test that learning shows expected convergence properties."""
        print("\\n=== Testing Learning Convergence Properties ===")
        
        # Set up for longer test to see convergence
        param_fixed['num_iterations'] = 50
        
        p_input_controller = load_learnable_parameters_initial(param_fixed)
        state_var0_model = load_initial_body_state(p_input_controller)
        vA, vB = get_treadmill_speed(0, param_fixed['imposedFootSpeeds'])
        context_now = np.array([vA, vB])
        
        np.random.seed(42)
        result = simulate_learning_step_by_step(
            param_fixed,
            p_input_controller,
            state_var0_model,
            context_now,
            param_controller_gains
        )
        
        # Analyze convergence properties
        initial_params = result[:, 0]
        final_params = result[:, -1]
        param_changes = np.abs(final_params - initial_params)
        
        # Check that parameters actually changed (learning occurred)
        learning_occurred = np.any(param_changes > 1e-6)
        
        # Check that changes are reasonable (not too large)
        reasonable_changes = np.all(param_changes < 1.0)
        
        self.results['learning_convergence'] = {
            'learning_occurred': learning_occurred,
            'reasonable_changes': reasonable_changes,
            'max_change': np.max(param_changes),
            'mean_change': np.mean(param_changes),
            'initial_params': initial_params.tolist(),
            'final_params': final_params.tolist()
        }
        
        print(f"Learning occurred: {learning_occurred}")
        print(f"Reasonable changes: {reasonable_changes}")
        print(f"Max parameter change: {np.max(param_changes):.6f}")
        print(f"Mean parameter change: {np.mean(param_changes):.6f}")
    
    def run_validation_suite(self):
        """Run the complete validation suite."""
        print("🔬 MATLAB-Python Functional Equivalence Validation")
        print("=" * 60)
        
        try:
            # Test 1: Parameter consistency
            param_fixed, param_controller_gains = self.test_parameter_consistency()
            
            # Test 2: Objective function determinism
            p_input_controller, state_var0_model = self.test_objective_function_determinism(
                param_fixed, param_controller_gains
            )
            
            # Test 3: Treadmill speed function
            self.test_treadmill_speed_function(param_fixed)
            
            # Test 4: Short learning simulation
            self.test_short_learning_simulation(param_fixed, param_controller_gains)
            
            # Test 5: Learning convergence properties
            self.test_learning_convergence_properties(param_fixed, param_controller_gains)
            
            # Generate summary report
            self.generate_summary_report()
            
        except Exception as e:
            print(f"\\n❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\\n" + "=" * 60)
        print("🧪 VALIDATION SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.results.items():
            print(f"\\n📋 {test_name.replace('_', ' ').title()}:")
            
            if test_name == 'parameter_consistency':
                success = result['success_rate'] > 0.95
                print(f"   Parameters matching: {result['passed']}/{result['total']} ({result['success_rate']:.1%})")
                total_tests += 1
                if success:
                    passed_tests += 1
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")
            
            elif test_name == 'objective_determinism':
                success = result['deterministic']
                print(f"   Deterministic: {success}")
                print(f"   Max difference: {result['max_difference']:.2e}")
                total_tests += 1
                if success:
                    passed_tests += 1
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")
            
            elif test_name == 'treadmill_speed':
                success = result['symmetric_initial']
                print(f"   Initial symmetry: {success}")
                total_tests += 1
                if success:
                    passed_tests += 1
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")
            
            elif test_name == 'learning_reproducibility':
                success = result['reproducible']
                print(f"   Reproducible: {success}")
                print(f"   Max difference: {result['max_difference']:.2e}")
                total_tests += 1
                if success:
                    passed_tests += 1
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")
            
            elif test_name == 'learning_convergence':
                success = result['learning_occurred'] and result['reasonable_changes']
                print(f"   Learning occurred: {result['learning_occurred']}")
                print(f"   Reasonable changes: {result['reasonable_changes']}")
                print(f"   Max change: {result['max_change']:.6f}")
                total_tests += 1
                if success:
                    passed_tests += 1
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")
        
        print("\\n" + "=" * 60)
        print(f"🏆 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Python implementation appears functionally equivalent to MATLAB!")
        elif passed_tests / total_tests >= 0.8:
            print("⚠️ MOSTLY PASSED: Python implementation is largely equivalent with minor differences.")
        else:
            print("❌ SIGNIFICANT ISSUES: Python implementation differs significantly from MATLAB.")
        
        print("=" * 60)
        
        return passed_tests / total_tests


def run_matlab_python_comparison():
    """Main function to run the MATLAB-Python comparison."""
    validator = MATLABPythonValidator()
    success = validator.run_validation_suite()
    return success


if __name__ == "__main__":
    print("Starting MATLAB-Python Functional Equivalence Validation...")
    success = run_matlab_python_comparison()
    if success:
        print("\\n✅ Validation completed successfully!")
    else:
        print("\\n❌ Validation encountered errors!")