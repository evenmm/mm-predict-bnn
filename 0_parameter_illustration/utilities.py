# -*- coding: utf-8 -*-
# 23.02.2024 Even Moa Myklebust

import numpy as np
import matplotlib.pyplot as plt

class Treatment:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id

class Parameters: 
    def __init__(self, Y_0, pi_r, g_r, g_s, k_1, sigma):
        self.Y_0 = Y_0 # M protein value at start of treatment
        self.pi_r = pi_r # Fraction of resistant cells at start of treatment 
        self.g_r = g_r # Growth rate of resistant cells
        self.g_s = g_s # Growth rate of sensitive cells in absence of treatment
        self.k_1 = k_1 # Additive effect of treatment on growth rate of sensitive cells
        self.sigma = sigma # Standard deviation of measurement noise
    def to_array_without_sigma(self):
        return np.array([self.Y_0, self.pi_r, self.g_r, self.g_s, self.k_1])
    def to_array_with_sigma(self):
        return np.array([self.Y_0, self.pi_r, self.g_r, self.g_s, self.k_1, self.sigma])
    def to_array_for_prediction(self):
        return np.array([self.pi_r, self.g_r, (self.g_s - self.k_1)])
    def to_prediction_array_composite_g_s_and_K_1(self):
        return [self.pi_r, self.g_r, (self.g_s - self.k_1)]

class Patient: 
    def __init__(self, parameters, measurement_times, treatment_history, covariates = [], name = "nn"):
        self.measurement_times = measurement_times
        self.treatment_history = treatment_history
        self.Mprotein_values = measure_Mprotein_with_noise(parameters, self.measurement_times, self.treatment_history)
        self.covariates = covariates
        self.name = name
        self.longitudinal_data = {} # dictionary where the keys are covariate_names, the entries numpy arrays! 
        self.longitudinal_times = {} # matching time arrays for each covariate
        self.pfs = []
        self.mrd_results = []
        self.mrd_times = []
    def get_measurement_times(self):
        return self.measurement_times
    def get_treatment_history(self):
        return self.treatment_history
    def get_Mprotein_values(self):
        return self.Mprotein_values
    def get_covariates(self):
        return self.covariates

#####################################
# Generative models, simulated data
#####################################
# Efficient implementation 
# Simulates M protein value at times [t + delta_T]_i
# Y_t is the M protein level at start of time interval
def generative_model(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values) + Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

def generate_resistant_Mprotein(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values)

def generate_sensitive_Mprotein(Y_t, params, delta_T_values, drug_effect):
    return Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

def get_pi_r_after_time_has_passed(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times, dtype=float) #without dtype you get ints
    # Adding a small epsilon to Y and pi_r to improve numerical stability
    epsilon_value = 1e-15
    Y_t = params.Y_0# + epsilon_value
    pi_r_t = params.pi_r# + epsilon_value
    t_params = Parameters(Y_t, pi_r_t, params.g_r, params.g_s, params.k_1, params.sigma)
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        this_treatment = treatment_history[treat_index]
        if this_treatment.id == 0:
            drug_effect = 0
        #elif this_treatment.id == 1:
        # With inference only for individual combinations at a time, it is either 0 or "treatment on", which is k1
        else:
            drug_effect = t_params.k_1
        #else:
        #    sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        
        # Filter that selects measurement times occuring while on this treatment line
        correct_times = (measurement_times >= this_treatment.start) & (measurement_times <= this_treatment.end)
        
        delta_T_values = measurement_times[correct_times] - this_treatment.start
        # Add delta T for (end - start) to keep track of Mprotein at end of treatment
        delta_T_values = np.concatenate((delta_T_values, np.array([this_treatment.end - this_treatment.start])))

        # Calculate Mprotein values
        # resistant 
        resistant_mprotein = generate_resistant_Mprotein(Y_t, t_params, delta_T_values, drug_effect)
        # sensitive
        sensitive_mprotein = generate_sensitive_Mprotein(Y_t, t_params, delta_T_values, drug_effect)
        # summed
        recorded_and_endtime_mprotein_values = resistant_mprotein + sensitive_mprotein
        # Assign M protein values for measurement times that are in this treatment period
        Mprotein_values[correct_times] = recorded_and_endtime_mprotein_values[0:-1]
        # Store Mprotein value at the end of this treatment:
        Y_t = recorded_and_endtime_mprotein_values[-1]# + epsilon_value
        pi_r_t = resistant_mprotein[-1] / (resistant_mprotein[-1] + sensitive_mprotein[-1] + epsilon_value) # Add a small number to keep numerics ok
        t_params = Parameters(Y_t, pi_r_t, t_params.g_r, t_params.g_s, t_params.k_1, t_params.sigma)
    return Mprotein_values, pi_r_t

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_noiseless(params, measurement_times, treatment_history):
    Mprotein_values, pi_r_after_time_has_passed = get_pi_r_after_time_has_passed(params, measurement_times, treatment_history)
    return Mprotein_values

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_with_noise(params, measurement_times, treatment_history):
    # Return true M protein value + Noise
    noise_array = np.random.normal(0, params.sigma, len(measurement_times))
    noisy_observations = measure_Mprotein_noiseless(params, measurement_times, treatment_history) + noise_array
    # thresholded at 0
    return np.array([max(0, value) for value in noisy_observations])
