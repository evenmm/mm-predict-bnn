import pickle
import sys
import time
import warnings
from copy import deepcopy
from multiprocessing import Pool

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import optimize
from scipy.optimize import least_squares

def isNaN(string):
    return string != string
def Sort(sub_li): # Sorts a list of sublists on the second element in the list 
    return(sorted(sub_li, key = lambda x: x[1]))
def find_max_time(measurement_times):
    # Plot until last measurement time (last in array, or first nan in array)
    if np.isnan(measurement_times).any():
        last_time_index = np.where(np.isnan(measurement_times))[0][0] -1 # Last non-nan index
    else:
        last_time_index = -1
    return int(measurement_times[last_time_index])

np.random.seed(42)
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))
# Later, you want to restart the stream
#rs = RandomState(MT19937(SeedSequence(987654321)))

# Assumptions and definitions: 
# Treatment lines must be back to back: Start of a treatment must equal end of previous treatment

#####################################
# Classes, functions
#####################################

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

class Treatment:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id

class Drug_period:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id

class Patient: 
    def __init__(self, parameters, measurement_times, treatment_history, covariates = [], name = "nn"):
        self.measurement_times = measurement_times
        self.treatment_history = treatment_history
        self.Mprotein_values = measure_Mprotein_with_noise(parameters, self.measurement_times, self.treatment_history)
        self.covariates = covariates
        self.name = name
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
    Mprotein_values = np.zeros_like(measurement_times)
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

# Pass a Parameter object to this function along with an numpy array of time points in days
def measure_Mprotein_naive(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times)
    Y_t = params.Y_0
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        if treatment_history[treat_index].id == 0:
            drug_effect = 0
        elif treatment_history[treat_index].id == 1:
            drug_effect = params.k_1
        else:
            print("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
            sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        # Calculate the M protein value at the end of this treatment line
        Mprotein_values = params.Y_0 * params.pi_r * np.exp(params.g_r * measurement_times) + params.Y_0 * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * measurement_times)        
    return Mprotein_values
    #return params.Y_0 * params.pi_r * np.exp(params.g_r * measurement_times) + params.Y_0 * (1-params.pi_r) * np.exp((params.g_s - params.k_1) * measurement_times)

def generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_local, P, get_expected_theta_from_X, true_omega, true_omega_for_psi, seed, RANDOM_EFFECTS, DIFFERENT_LENGTHS=False, min_length=5):
    np.random.seed(seed)
    #X_mean = np.repeat(0,P)
    #X_std = np.repeat(0.5,P)
    #X = np.random.normal(X_mean, X_std, size=(N_patients_local,P))
    X = np.random.uniform(-1, 1, size=(N_patients_local,P))
    X = pd.DataFrame(X, columns = ["Covariate "+str(ii+1) for ii in range(P)])

    expected_theta_1, expected_theta_2, expected_theta_3 = get_expected_theta_from_X(X)

    # Set the seed again to make the random effects not change with P
    np.random.seed(seed+1)
    if RANDOM_EFFECTS:
        true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
        true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
        true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])
    else:
        true_theta_rho_s = expected_theta_1
        true_theta_rho_r = expected_theta_2
        true_theta_pi_r  = expected_theta_3

    # Set the seed again to get identical observation noise irrespective of random effects or not
    np.random.seed(seed+2)
    psi_population = 50
    true_theta_psi = np.random.normal(np.log(psi_population), true_omega_for_psi, size=N_patients_local)
    true_rho_s = - np.exp(true_theta_rho_s)
    true_rho_r = np.exp(true_theta_rho_r)
    true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
    true_psi = np.exp(true_theta_psi)

    # Set seed again to give patient random Numbers of M protein
    np.random.seed(seed+3)
    parameter_dictionary = {}
    patient_dictionary = {}
    for training_instance_id in range(N_patients_local):
        psi_patient_i   = true_psi[training_instance_id]
        pi_r_patient_i  = true_pi_r[training_instance_id]
        rho_r_patient_i = true_rho_r[training_instance_id]
        rho_s_patient_i = true_rho_s[training_instance_id]
        these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=true_sigma_obs)
        if DIFFERENT_LENGTHS:
            # Remove some measurement times from the end: 
            #M_ii = np.random.randint(min(3,len(measurement_times)), len(measurement_times)+1)
            M_ii = np.random.randint(min(min_length,len(measurement_times)), len(measurement_times)+1)
            measurement_times_ii = measurement_times[:M_ii]
        else:
            measurement_times_ii = measurement_times
        this_patient = Patient(these_parameters, measurement_times_ii, treatment_history, name=str(training_instance_id))
        patient_dictionary[training_instance_id] = this_patient
        parameter_dictionary[training_instance_id] = these_parameters
        #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id)
    return X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r

#####################################
# Plotting
#####################################
#treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
def plot_mprotein(patient, title, savename):
    measurement_times = patient.measurement_times
    Mprotein_values = patient.Mprotein_values
    
    fig, ax1 = plt.subplots()
    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")

    ax1.set_title(title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0)
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    plt.close()

def plot_true_mprotein_with_observations_and_treatments_and_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    first_time = min(measurement_times[0], treatment_history[0].start)
    max_time = find_max_time(measurement_times)
    plotting_times = np.linspace(first_time, max_time, int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    # Plot sensitive and resistant
    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="True M protein (resistant)")
    # Plot total M protein
    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', label="True M protein (total)")
    if PLOT_ESTIMATES:
        # Plot estimtated Mprotein line
        estimated_mprotein_values = measure_Mprotein_noiseless(estimated_parameters, plotting_times, treatment_history)
        ax1.plot(plotting_times, estimated_mprotein_values, linestyle='--', linewidth=2, marker='', zorder=3, color='b', label="Estimated M protein")

    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        # Adaptation made to simulation study 2:
        if treat_index>0:
            ax1.axvline(this_treatment.start, color="k", linewidth=1, linestyle="-", label="Start of period of interest")
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start
            if this_treatment.id > maxdrugkey:
                maxdrugkey = this_treatment.id

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary_OSLO[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("Treatment line. max="+str(maxdrugkey))
    ax2.set_yticks(range(maxdrugkey+1))
    ax2.set_yticklabels(range(maxdrugkey+1))
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    if savename != 0:
        plt.savefig(savename,dpi=300)
    else:
        if PLOT_ESTIMATES:
            plt.savefig("./patient_truth_and_observations_with_model_fit"+plot_title+".pdf",dpi=300)
        else:
            plt.savefig("./patient_truth_and_observations"+plot_title+".pdf",dpi=300)
    #plt.show()
    plt.close()

def plot_treatment_region_with_estimate(true_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 0

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    # Plot sensitive and resistant
    if true_parameters.pi_r > 10e-10 and true_parameters.pi_r < 1-10e-10: 
        # Plot resistant
        ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")
    elif true_parameters.pi_r > 1-10e-10:
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='r', label="Estimated M protein (total)")
    else:
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")

    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")
    #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start
            if this_treatment.id > maxdrugkey:
                maxdrugkey = this_treatment.id

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary_OSLO[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    ax2.set_ylabel("Treatment id for blue region")
    ax2.set_yticks([maxdrugkey])
    ax2.set_yticklabels([maxdrugkey])
    ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    #plt.show()
    plt.close()


def plot_to_compare_estimated_and_predicted_drug_dynamics(true_parameters, predicted_parameters, patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title="Patient 1", savename=0):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, int((measurement_times[-1]+1)*10))
    
    # Plot true M protein values according to true parameters
    plotting_mprotein_values = measure_Mprotein_noiseless(true_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters = Parameters((true_parameters.Y_0*true_parameters.pi_r), 1, true_parameters.g_r, true_parameters.g_s, true_parameters.k_1, true_parameters.sigma)
    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)

    # Plot predicted M protein values according to predicted parameters
    plotting_mprotein_values_pred = measure_Mprotein_noiseless(predicted_parameters, plotting_times, treatment_history)
    # Count resistant part
    resistant_parameters_pred = Parameters((predicted_parameters.Y_0*predicted_parameters.pi_r), 1, predicted_parameters.g_r, predicted_parameters.g_s, predicted_parameters.k_1, predicted_parameters.sigma)
    plotting_resistant_mprotein_values_pred = measure_Mprotein_noiseless(resistant_parameters_pred, plotting_times, treatment_history)

    # Plot M protein values
    plotheight = 1
    maxdrugkey = 2

    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')
    # Plot sensitive and resistant
    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
    # Plot sensitive and resistant
    ax1.plot(plotting_times, plotting_resistant_mprotein_values_pred, linestyle='--', marker='', zorder=3, color='orange', label="Predicted M protein (resistant)")
    # Plot total M protein
    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")
    # Plot total M protein, predicted
    ax1.plot(plotting_times, plotting_mprotein_values_pred, linestyle='--', marker='', zorder=3, color='b', label="Predicted M protein (total)")

    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")
    [ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    ax2 = ax1.twinx() 
    for treat_index in range(len(treatment_history)):
        this_treatment = treatment_history[treat_index]
        if this_treatment.id != 0:
            treatment_duration = this_treatment.end - this_treatment.start
            if this_treatment.id > maxdrugkey:
                maxdrugkey = this_treatment.id

            #drugs_1 = list of drugs from dictionary mapping id-->druglist, key=this_treatment.id
            #for ii in range(len(drugs_1)):
            #    drugkey = drug_dictionary_OSLO[drugs_1[ii]]
            #    if drugkey > maxdrugkey:
            #        maxdrugkey = drugkey
            #    #             Rectangle(             x                   y            ,        width      ,   height  , ...)
            #    ax2.add_patch(Rectangle((this_treatment.start, drugkey - plotheight/2), treatment_duration, plotheight, zorder=2, color=drug_colordict[drugkey]))
            ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    ax2.set_ylabel("Treatment line. max="+str(maxdrugkey))
    ax2.set_yticks(range(maxdrugkey+1))
    ax2.set_yticklabels(range(maxdrugkey+1))
    ax2.set_ylim(bottom=0, top=maxdrugkey+plotheight/2)
    #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    #plt.show()
    plt.close()

# Plot posterior confidence intervals 
def plot_posterior_confidence_intervals(training_instance_id, patient, sorted_pred_y_values, parameter_estimates=[], PLOT_POINT_ESTIMATES=False, PLOT_TREATMENTS=False, plot_title="", savename="0", y_resolution=1000, n_chains=4, n_samples=1000):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, y_resolution)
    
    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')

    if PLOT_POINT_ESTIMATES:
        # Plot true M protein values according to parameter estimates
        plotting_mprotein_values = measure_Mprotein_noiseless(parameter_estimates, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameter_estimates.Y_0*parameter_estimates.pi_r), 1, parameter_estimates.g_r, parameter_estimates.g_s, parameter_estimates.k_1, parameter_estimates.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # Plot resistant M protein
        ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='-', marker='', zorder=3, color='r', label="Estimated M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='k', label="Estimated M protein (total)")

    # Plot posterior confidence intervals 
    # 95 % empirical confidence interval
    color_array = ["#fbd1b4", "#f89856", "#e36209"] #["#fbd1b4", "#fab858", "#f89856", "#f67c27", "#e36209"] #https://icolorpalette.com/color/rust-orange
    for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
        # Get index to find right value 
        lower_index = int(critical_value*sorted_pred_y_values.shape[0]) #n_chains*n_samples)
        upper_index = int((1-critical_value)*sorted_pred_y_values.shape[0]) #n_chains*n_samples)
        # index at intervals to get 95 % limit value
        lower_limits = sorted_pred_y_values[lower_index,training_instance_id,:]
        upper_limits = sorted_pred_y_values[upper_index,training_instance_id,:]       #color=color_array[index]
        ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.copper(1-critical_value), label='%3.0f %% confidence band on M protein value' % (100*(1-2*critical_value)))

    # Plot M protein observations
    ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein") #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    if PLOT_TREATMENTS:
        plotheight = 1
        maxdrugkey = 0
        ax2 = ax1.twinx() 
        for treat_index in range(len(treatment_history)):
            this_treatment = treatment_history[treat_index]
            if this_treatment.id != 0:
                treatment_duration = this_treatment.end - this_treatment.start
                if this_treatment.id > maxdrugkey:
                    maxdrugkey = this_treatment.id
                ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=2, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    if PLOT_TREATMENTS:
        ax2.set_ylabel("Treatment id for blue region")
        ax2.set_yticks([maxdrugkey])
        ax2.set_yticklabels([maxdrugkey])
        ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    #plt.show()
    plt.close()

def plot_posterior_local_confidence_intervals(training_instance_id, patient, sorted_local_pred_y_values, parameters=[], PLOT_PARAMETERS=False, PLOT_TREATMENTS=False, plot_title="", savename="0", y_resolution=1000, n_chains=4, n_samples=1000, sorted_resistant_mprotein=[], PLOT_MEASUREMENTS = True, PLOT_RESISTANT=True):
    measurement_times = patient.get_measurement_times()
    treatment_history = patient.get_treatment_history()
    Mprotein_values = patient.get_Mprotein_values()
    time_zero = min(treatment_history[0].start, measurement_times[0])
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(time_zero, time_max, y_resolution)
    
    fig, ax1 = plt.subplots()
    ax1.patch.set_facecolor('none')

    # Plot posterior confidence intervals for Resistant M protein
    # 95 % empirical confidence interval
    if PLOT_RESISTANT:
        if len(sorted_resistant_mprotein) > 0: 
            for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
                # Get index to find right value 
                lower_index = int(critical_value*sorted_resistant_mprotein.shape[0]) #n_chains*n_samples)
                upper_index = int((1-critical_value)*sorted_resistant_mprotein.shape[0]) #n_chains*n_samples)
                # index at intervals to get 95 % limit value
                lower_limits = sorted_resistant_mprotein[lower_index,:]
                upper_limits = sorted_resistant_mprotein[upper_index,:]
                ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.copper(1-critical_value), label='%3.0f %% conf. for resistant M prot.' % (100*(1-2*critical_value)), zorder=0+index*0.1)

    # Plot posterior confidence intervals for total M protein
    # 95 % empirical confidence interval
    color_array = ["#fbd1b4", "#f89856", "#e36209"] #["#fbd1b4", "#fab858", "#f89856", "#f67c27", "#e36209"] #https://icolorpalette.com/color/rust-orange
    for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
        # Get index to find right value 
        lower_index = int(critical_value*sorted_local_pred_y_values.shape[0]) #n_chains*n_samples)
        upper_index = int((1-critical_value)*sorted_local_pred_y_values.shape[0]) #n_chains*n_samples)
        # index at intervals to get 95 % limit value
        lower_limits = sorted_local_pred_y_values[lower_index,:]
        upper_limits = sorted_local_pred_y_values[upper_index,:]
        shade_array = [0.7, 0.5, 0.35]
        ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.bone(shade_array[index]), label='%3.0f %% conf. for M prot. value' % (100*(1-2*critical_value)), zorder=1+index*0.1)

    if PLOT_PARAMETERS:
        # Plot true M protein curves according to parameters
        plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # Plot resistant M protein
        if PLOT_RESISTANT:
            ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color=plt.cm.hot(0.2), label="True M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='cyan', label="True M protein (total)")

    # Plot M protein observations
    if PLOT_MEASUREMENTS == True:
        ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=4, color='k', label="Observed M protein") #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    if PLOT_TREATMENTS:
        plotheight = 1
        maxdrugkey = 0
        ax2 = ax1.twinx()
        for treat_index in range(len(treatment_history)):
            this_treatment = treatment_history[treat_index]
            if this_treatment.id != 0:
                treatment_duration = this_treatment.end - this_treatment.start
                if this_treatment.id > maxdrugkey:
                    maxdrugkey = this_treatment.id
                ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=0, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    ax1.set_title(plot_title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum Mprotein (g/L)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    #ax1.set_xlim(left=time_zero)
    if PLOT_TREATMENTS:
        ax2.set_ylabel("Treatment id for blue region")
        ax2.set_yticks([maxdrugkey])
        ax2.set_yticklabels([maxdrugkey])
        ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    #handles, labels = ax1.get_legend_handles_labels()
    #lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    #ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename, dpi=300) #, bbox_extra_artists=(lgd), bbox_inches='tight')
    plt.close()

def plot_posterior_traces(idata, SAVEDIR, name, psi_prior, model_name, patientwise=True):
    if model_name == "linear":
        print("Plotting posterior/trace plots")
        # Autocorrelation plots: 
        az.plot_autocorr(idata, var_names=["sigma_obs"])

        az.plot_trace(idata, var_names=('alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma_obs'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_group_parameters.pdf")
        plt.close()

        az.plot_trace(idata, var_names=('beta_rho_s'), lines=[('beta_rho_s', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_rho_s.pdf")
        plt.close()

        # Combined means combine the chains into one posterior. Compact means split into different subplots
        az.plot_trace(idata, var_names=('beta_rho_r'), lines=[('beta_rho_r', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_rho_r.pdf")
        plt.close()

        # Combined means combine the chains into one posterior. Compact means split into different subplots
        az.plot_trace(idata, var_names=('beta_pi_r'), lines=[('beta_pi_r', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_pi_r.pdf")
        plt.close()

        az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-forest_beta_rho_r.pdf")
        plt.close()
        az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.savefig(SAVEDIR+name+"-forest_pi_r.pdf")
        plt.tight_layout()
        plt.close()
        az.plot_forest(idata, var_names=["beta_rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-forest_beta_rho_s.pdf")
        plt.close()
    elif model_name == "BNN":
        # Plot weights in_1 rho_s
        az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_s.pdf")
        plt.close()
        # Plot weights in_1 rho_r
        az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_r.pdf")
        plt.close()
        # Plot weights in_1 pi_r
        az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_pi_r.pdf")
        plt.close()

        # Plot weights 2_out rho_s
        az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_rho_s.pdf")
        plt.close()
        # Plot weights 2_out rho_r
        az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_rho_r.pdf")
        plt.close()
        # Plot weights 2_out pi_r
        az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_pi_r.pdf")
        plt.close()

        # Combined means combined chains
        # Plot weights in_1 rho_s
        az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_s_combined.pdf")
        plt.close()
        # Plot weights in_1 rho_r
        az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_r_combined.pdf")
        plt.close()
        # Plot weights in_1 pi_r
        az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_pi_r_combined.pdf")
        plt.close()

        # Plot weights 2_out rho_s
        az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_rho_s_combined.pdf")
        plt.close()
        # Plot weights 2_out rho_r
        az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_rho_r_combined.pdf")
        plt.close()
        # Plot weights 2_out pi_r
        az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_pi_r_combined.pdf")
        plt.close()
    elif model_name == "joint_BNN":
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out.pdf")
        plt.close()

        # Combined means combined chains
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_combined.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_combined.pdf")
        plt.close()

    if psi_prior=="lognormal":
        az.plot_trace(idata, var_names=('xi'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_group_parameters_xi.pdf")
        plt.close()
    # Test of exploration 
    az.plot_energy(idata)
    plt.savefig(SAVEDIR+name+"-plot_energy.pdf")
    plt.close()
    # Plot of coefficients
    az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_alpha.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_rho_s.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_rho_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_pi_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["psi"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_psi.pdf")
    plt.close()
    if patientwise:
        az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_individual_parameters.pdf")
        plt.close()

def plot_posterior_CI(args):
    sample_shape, y_resolution, ii, idata, patient_dictionary, SAVEDIR, name, N_rand_obs_pred_train, model_name, parameter_dictionary, PLOT_PARAMETERS, CI_with_obs_noise, PLOT_RESISTANT = args
    if not CI_with_obs_noise:
        N_rand_obs_pred_train = 1
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    np.random.seed(ii) # Seeding the randomness in observation noise sigma

    patient = patient_dictionary[ii]
    measurement_times = patient.get_measurement_times() 
    treatment_history = patient.get_treatment_history()
    first_time = min(measurement_times[0], treatment_history[0].start)
    time_max = find_max_time(measurement_times)
    plotting_times = np.linspace(first_time, time_max, y_resolution) #int((measurement_times[-1]+1)*10))
    posterior_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
    predicted_y_values = np.empty(shape=(n_chains, n_samples*N_rand_obs_pred_train, y_resolution))
    predicted_y_resistant_values = np.empty_like(predicted_y_values)
    for ch in range(n_chains):
        for sa in range(n_samples):
            this_sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
            this_psi       = np.ravel(idata.posterior['psi'][ch,sa,ii])
            this_pi_r      = np.ravel(idata.posterior['pi_r'][ch,sa,ii])
            this_rho_s     = np.ravel(idata.posterior['rho_s'][ch,sa,ii])
            this_rho_r     = np.ravel(idata.posterior['rho_r'][ch,sa,ii])
            posterior_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=this_pi_r, g_r=this_rho_r, g_s=this_rho_s, k_1=0, sigma=this_sigma_obs)
            these_parameters = posterior_parameters[ch,sa]
            resistant_parameters = Parameters((these_parameters.Y_0*these_parameters.pi_r), 1, these_parameters.g_r, these_parameters.g_s, these_parameters.k_1, these_parameters.sigma)
            # Predicted total and resistant M protein
            predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, plotting_times, treatment_history)
            predicted_y_resistant_values_noiseless = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
            # Add noise and make the resistant part the estimated fraction of the observed value
            if CI_with_obs_noise:
                for rr in range(N_rand_obs_pred_train):
                    noise_array = np.random.normal(0, this_sigma_obs, y_resolution)
                    noisy_observations = predicted_y_values_noiseless + noise_array
                    predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                    predicted_y_resistant_values[ch, N_rand_obs_pred_train*sa + rr] = predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
            else: 
                predicted_y_values[ch, sa] = predicted_y_values_noiseless
                predicted_y_resistant_values[ch, sa] = predicted_y_values[ch, sa] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
    flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*n_samples*N_rand_obs_pred_train,y_resolution))
    sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
    flat_pred_resistant = np.reshape(predicted_y_resistant_values, (n_chains*n_samples*N_rand_obs_pred_train,y_resolution))
    sorted_pred_resistant = np.sort(flat_pred_resistant, axis=0)
    savename = SAVEDIR+"CI_training_id_"+str(ii)+"_"+name+".pdf"
    if PLOT_PARAMETERS and len(parameter_dictionary) > 0:
        parameters_ii = parameter_dictionary[ii]
    else: 
        parameters_ii = []
    plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=parameters_ii, PLOT_PARAMETERS=PLOT_PARAMETERS, PLOT_TREATMENTS=False, plot_title="Posterior CI for training patient "+str(ii), savename=savename, y_resolution=y_resolution, n_chains=n_chains, n_samples=n_samples, sorted_resistant_mprotein=sorted_pred_resistant, PLOT_RESISTANT=PLOT_RESISTANT)
    return 0 # {"posterior_parameters" : posterior_parameters, "predicted_y_values" : predicted_y_values, "predicted_y_resistant_values" : predicted_y_resistant_values}

def plot_predictions(args): # Predicts observations of M protein
    #sample_shape, y_resolution, ii = args
    sample_shape, y_resolution, ii, idata, X_test, patient_dictionary_test, SAVEDIR, name, N_rand_eff_pred, N_rand_obs_pred, model_name, parameter_dictionary, PLOT_PARAMETERS, PLOT_TREATMENTS, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, PLOT_RESISTANT, PLOT_MEASUREMENTS = args
    if not CI_with_obs_noise:
        N_rand_eff_pred = N_rand_eff_pred * N_rand_obs_pred
        N_rand_obs_pred = 1
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    np.random.seed(ii) # Seeding the randomness in observation noise sigma, in random effects and in psi = yi0 + random(sigma)

    patient = patient_dictionary_test[ii]
    measurement_times = patient.get_measurement_times() 
    treatment_history = patient.get_treatment_history()
    first_time = min(measurement_times[0], treatment_history[0].start)
    max_time = find_max_time(measurement_times)
    plotting_times = np.linspace(first_time, max_time, y_resolution) #int((measurement_times[-1]+1)*10))
    predicted_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
    predicted_y_values = np.empty(shape=(n_chains*N_rand_eff_pred, n_samples*N_rand_obs_pred, y_resolution))
    predicted_y_resistant_values = np.empty_like(predicted_y_values)
    for ch in range(n_chains):
        for sa in range(n_samples):
            sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
            alpha = np.ravel(idata.posterior['alpha'][ch,sa])

            if model_name == "linear": 
                this_beta_rho_s = np.ravel(idata.posterior['beta_rho_s'][ch,sa])
                this_beta_rho_r = np.ravel(idata.posterior['beta_rho_r'][ch,sa])
                this_beta_pi_r = np.ravel(idata.posterior['beta_pi_r'][ch,sa])
            elif model_name == "BNN": 
                # weights 
                weights_in_rho_s = idata.posterior['weights_in_rho_s'][ch,sa]
                weights_in_rho_r = idata.posterior['weights_in_rho_r'][ch,sa]
                weights_in_pi_r = idata.posterior['weights_in_pi_r'][ch,sa]
                weights_out_rho_s = idata.posterior['weights_out_rho_s'][ch,sa]
                weights_out_rho_r = idata.posterior['weights_out_rho_r'][ch,sa]
                weights_out_pi_r = idata.posterior['weights_out_pi_r'][ch,sa]

                # intercepts
                #sigma_bias_in = idata.posterior['sigma_bias_in'][ch,sa]
                bias_in_rho_s = np.ravel(idata.posterior['bias_in_rho_s'][ch,sa])
                bias_in_rho_r = np.ravel(idata.posterior['bias_in_rho_r'][ch,sa])
                bias_in_pi_r = np.ravel(idata.posterior['bias_in_pi_r'][ch,sa])

                pre_act_1_rho_s = np.dot(X_test.iloc[ii,:], weights_in_rho_s) + bias_in_rho_s
                pre_act_1_rho_r = np.dot(X_test.iloc[ii,:], weights_in_rho_r) + bias_in_rho_r
                pre_act_1_pi_r  = np.dot(X_test.iloc[ii,:], weights_in_pi_r)  + bias_in_pi_r

                act_1_rho_s = np.select([pre_act_1_rho_s > 0, pre_act_1_rho_s <= 0], [pre_act_1_rho_s, pre_act_1_rho_s*0.01], 0)
                act_1_rho_r = np.select([pre_act_1_rho_r > 0, pre_act_1_rho_r <= 0], [pre_act_1_rho_r, pre_act_1_rho_r*0.01], 0)
                act_1_pi_r =  np.select([pre_act_1_pi_r  > 0, pre_act_1_pi_r  <= 0], [pre_act_1_pi_r,  pre_act_1_pi_r*0.01],  0)

                # Output
                act_out_rho_s = np.dot(act_1_rho_s, weights_out_rho_s)
                act_out_rho_r = np.dot(act_1_rho_r, weights_out_rho_r)
                act_out_pi_r =  np.dot(act_1_pi_r,  weights_out_pi_r)

            elif model_name == "joint_BNN": 
                # weights 
                weights_in = idata.posterior['weights_in'][ch,sa]
                weights_out = idata.posterior['weights_out'][ch,sa]

                # intercepts
                #sigma_bias_in = idata.posterior['sigma_bias_in'][ch,sa]
                bias_in = np.ravel(idata.posterior['bias_in'][ch,sa])

                pre_act_1 = np.dot(X_test.iloc[ii,:], weights_in) + bias_in

                act_1 = np.select([pre_act_1 > 0, pre_act_1 <= 0], [pre_act_1, pre_act_1*0.01], 0)

                # Output
                act_out = np.dot(act_1, weights_out)
                act_out_rho_s = act_out[0]
                act_out_rho_r = act_out[1]
                act_out_pi_r =  act_out[2]

            # Random effects 
            omega  = np.ravel(idata.posterior['omega'][ch,sa])
            for ee in range(N_rand_eff_pred):
                if model_name == "linear":
                    #if MODEL_RANDOM_EFFECTS: 
                    predicted_theta_1 = np.random.normal(alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s), omega[0])
                    predicted_theta_2 = np.random.normal(alpha[1] + np.dot(X_test.iloc[ii,:], this_beta_rho_r), omega[1])
                    predicted_theta_3 = np.random.normal(alpha[2] + np.dot(X_test.iloc[ii,:], this_beta_pi_r), omega[2])
                    #else: 
                    #    predicted_theta_1 = alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s)
                    #    predicted_theta_2 = alpha[1] + np.dot(X_test.iloc[ii,:], this_beta_rho_r)
                    #    predicted_theta_3 = alpha[2] + np.dot(X_test.iloc[ii,:], this_beta_pi_r)
                elif model_name == "BNN" or model_name == "joint_BNN":
                    if MODEL_RANDOM_EFFECTS:
                        predicted_theta_1 = np.random.normal(alpha[0] + act_out_rho_s, omega[0])
                        predicted_theta_2 = np.random.normal(alpha[1] + act_out_rho_r, omega[1])
                        predicted_theta_3 = np.random.normal(alpha[2] + act_out_pi_r, omega[2])
                    else: 
                        predicted_theta_1 = alpha[0] + act_out_rho_s
                        predicted_theta_2 = alpha[1] + act_out_rho_r
                        predicted_theta_3 = alpha[2] + act_out_pi_r

                predicted_rho_s = - np.exp(predicted_theta_1)
                predicted_rho_r = np.exp(predicted_theta_2)
                predicted_pi_r  = 1/(1+np.exp(-predicted_theta_3))

                this_psi = patient.Mprotein_values[0] + np.random.normal(0,sigma_obs)
                predicted_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=predicted_pi_r, g_r=predicted_rho_r, g_s=predicted_rho_s, k_1=0, sigma=sigma_obs)
                these_parameters = predicted_parameters[ch,sa]
                resistant_parameters = Parameters(Y_0=(these_parameters.Y_0*these_parameters.pi_r), pi_r=1, g_r=these_parameters.g_r, g_s=these_parameters.g_s, k_1=these_parameters.k_1, sigma=these_parameters.sigma)
                # Predicted total and resistant M protein
                predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, plotting_times, treatment_history)
                predicted_y_resistant_values_noiseless = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Add noise and make the resistant part the estimated fraction of the observed value
                if CI_with_obs_noise:
                    for rr in range(N_rand_obs_pred):
                        noise_array = np.random.normal(0, sigma_obs, y_resolution)
                        noisy_observations = predicted_y_values_noiseless + noise_array
                        predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                        predicted_y_resistant_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
                else: 
                    predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = predicted_y_values_noiseless
                    predicted_y_resistant_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
    flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*n_samples*N_rand_eff_pred*N_rand_obs_pred,y_resolution))
    sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
    flat_pred_resistant = np.reshape(predicted_y_resistant_values, (n_chains*n_samples*N_rand_eff_pred*N_rand_obs_pred,y_resolution))
    sorted_pred_resistant = np.sort(flat_pred_resistant, axis=0)
    savename = SAVEDIR+"CI_test_id_"+str(ii)+"_"+name+".pdf"
    if PLOT_PARAMETERS and len(parameter_dictionary) > 0:
        parameters_ii = parameter_dictionary[ii]
    else:
        parameters_ii = []
    plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=parameters_ii, PLOT_PARAMETERS=PLOT_PARAMETERS, plot_title="Posterior predictive CI for test patient "+str(ii), savename=savename, y_resolution=y_resolution, n_chains=n_chains, n_samples=n_samples, sorted_resistant_mprotein=sorted_pred_resistant, PLOT_MEASUREMENTS = PLOT_MEASUREMENTS, PLOT_RESISTANT=PLOT_RESISTANT)
    return 0 # {"posterior_parameters" : posterior_parameters, "predicted_y_values" : predicted_y_values, "predicted_y_resistant_values" : predicted_y_resistant_values}

def plot_all_credible_intervals(idata, patient_dictionary, patient_dictionary_test, X_test, SAVEDIR, name, y_resolution, model_name, parameter_dictionary, PLOT_PARAMETERS, parameter_dictionary_test, PLOT_PARAMETERS_test, PLOT_TREATMENTS, MODEL_RANDOM_EFFECTS, CI_with_obs_noise=True, PARALLELLIZE=True, PLOT_RESISTANT=True, PLOT_MEASUREMENTS_test=False):
    sample_shape = idata.posterior['psi'].shape # [chain, n_samples, dim]
    N_chains = sample_shape[0]
    N_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient

    # Posterior CI for train data
    if N_samples <= 10:
        N_rand_obs_pred_train = 10000 # Number of observation noise samples to draw for each parameter sample
    elif N_samples <= 100:
        N_rand_obs_pred_train = 1000 # Number of observation noise samples to draw for each parameter sample
    elif N_samples <= 1000:
        N_rand_obs_pred_train = 100 # Number of observation noise samples to draw for each parameter sample
    else:
        N_rand_obs_pred_train = 10 # Number of observation noise samples to draw for each parameter sample
    print("Plotting posterior credible bands for training cases")
    N_patients = len(patient_dictionary)
    args = [(sample_shape, y_resolution, ii, idata, patient_dictionary, SAVEDIR, name, N_rand_obs_pred_train, model_name, parameter_dictionary, PLOT_PARAMETERS, CI_with_obs_noise, PLOT_RESISTANT) for ii in range(min(N_patients, 15))]
    if PARALLELLIZE:
        if SAVEDIR in ["./plots/Bayesian_estimates_simdata_linearmodel/", "./plots/Bayesian_estimates_simdata_BNN/", "./plots/Bayesian_estimates_simdata_joint_BNN/"]:
            poolworkers = 15
        else:
            poolworkers = 4 
        with Pool(poolworkers) as pool:
            results = pool.map(plot_posterior_CI,args)
    else: 
        for elem in args:
            plot_posterior_CI(elem)
    print("...done.")

    # Posterior predictive CI for test data
    if N_samples <= 10:
        N_rand_eff_pred = 100 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 100:
        N_rand_eff_pred = 10 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    elif N_samples <= 1000:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
    else:
        N_rand_eff_pred = 1 # Number of random intercept samples to draw for each idata sample when we make predictions
        N_rand_obs_pred = 10 # Number of observation noise samples to draw for each parameter sample 
    print("Plotting predictive credible bands for test cases")
    N_patients_test = len(patient_dictionary_test)
    args = [(sample_shape, y_resolution, ii, idata, X_test, patient_dictionary_test, SAVEDIR, name, N_rand_eff_pred, N_rand_obs_pred, model_name, parameter_dictionary_test, PLOT_PARAMETERS_test, PLOT_TREATMENTS, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, PLOT_RESISTANT, PLOT_MEASUREMENTS_test) for ii in range(min(N_patients_test, 15))]
    if PARALLELLIZE:
        with Pool(poolworkers) as pool:
            results = pool.map(plot_predictions,args)
    else: 
        for elem in args:
            plot_predictions(elem)
    print("...done.")

def plot_parameter_dependency_on_covariates(SAVEDIR, name, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r):
    color_array = X["Covariate 2"].to_numpy()
    striplets = [("expected_theta_1", "true_theta_rho_s", "true_rho_s"), ("expected_theta_2", "true_theta_rho_r", "true_rho_r"), ("expected_theta_3", "true_theta_pi_r", "true_pi_r")]
    for ii, triplet in enumerate([(expected_theta_1, true_theta_rho_s, true_rho_s), (expected_theta_2, true_theta_rho_r, true_rho_r), (expected_theta_3, true_theta_pi_r, true_pi_r)]):
        striplet = striplets[ii]
        for jj, elem in enumerate(triplet):
            selem = striplet[jj]
            for covariate in ["Covariate 1", "Covariate 3"]:
                # Covariates 1 and 2
                fig, ax = plt.subplots()
                ax.set_title(selem)
                points = ax.scatter(X[covariate], elem, c=color_array, cmap="plasma")
                ax.set_xlabel(covariate)
                ax.set_ylabel(selem)
                cbar = fig.colorbar(points)
                cbar.set_label('covariate 2', rotation=90)
                plt.savefig(SAVEDIR + "_".join(["effects",covariate,str(ii),str(jj),selem,name,".pdf"]), dpi=300)
                plt.close()

#####################################
# Posterior evaluation
#####################################
# Convergence checks
def quasi_geweke_test(idata, model_name, first=0.1, last=0.5, intervals=20):
    if first+last > 1:
        print("Overriding input since first+last>1. New first, last = 0.1, 0.5")
        first, last = 0.1, 0.5
    print("Running Geweke test...")
    convergence_flag = True
    if model_name == "linear":
        var_names = ['alpha', 'beta_rho_s', 'beta_rho_r', 'beta_pi_r', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    elif model_name == "BNN":
        var_names = ['alpha', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    for var_name in var_names:
        sample_shape = idata.posterior[var_name].shape
        n_chains = sample_shape[0]
        n_samples = sample_shape[1]
        var_dims = sample_shape[2]
        for chain in range(n_chains):
            for dim in range(var_dims):
                all_samples = np.ravel(idata.posterior[var_name][chain,:,dim])
                first_part = all_samples[0:int(n_samples*first)]
                last_part = all_samples[n_samples-int(n_samples*last):n_samples]
                z_score = (np.mean(first_part)-np.mean(last_part)) / np.sqrt(np.var(first_part)+np.var(last_part))
                if abs(z_score) >= 1.960:
                    convergence_flag = False
                    #print("Seems like chain",chain,"has not converged in",var_name,"dimension",dim,": z_score is",z_score)
    for var_name in ['sigma_obs']:
        all_samples = np.ravel(idata.posterior[var_name])
        n_samples = len(all_samples)
        first_part = all_samples[0:int(n_samples*first)]
        last_part = all_samples[n_samples-int(n_samples*last):n_samples]
        z_score = (np.mean(first_part)-np.mean(last_part)) / np.sqrt(np.var(first_part)+np.var(last_part))
        if abs(z_score) >= 1.960:
            convergence_flag = False
            print("Seems like chain",chain,"has not converged in",var_name,"dimension",dim,": z_score is",z_score)
    if convergence_flag:
        print("All chains seem to have converged.")
    else:
        print("Seems like some chains did not converge.")
    return 0

################################
# Data simulation
################################

# Function to get expected theta from X
def get_expected_theta_from_X_one_interaction(X): # One interaction: In rho_s only
    # These are the true parameters for a patient with all covariates equal to 0:
    N_patients_local, P = X.shape
    rho_s_population = -0.005
    rho_r_population = 0.001
    pi_r_population = 0.3
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    true_beta_rho_s[0] = 0.8
    true_beta_rho_s[1] = 0
    interaction_beta_x1_x2_rho_s = -1
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 1.0
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.0
    true_beta_pi_r[1] = 1.1

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s) + np.ravel(interaction_beta_x1_x2_rho_s*X["Covariate 1"]*(X["Covariate 2"].T)), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients_local,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3

def get_expected_theta_from_X_2(X): # One interaction: In rho_s only
    N_patients_local, P = X.shape
    # These are the true parameters for a patient with all covariates equal to 0:
    rho_s_population = -0.04 #-0.02
    rho_r_population = 0.008 #0.004
    pi_r_population = 0.2 #0.3
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    #true_beta_rho_s[0] = 0.8
    #true_beta_rho_s[1] = 0
    #true_beta_rho_s[2] = 0.4
    #true_beta_rho_s[3] = 0.3
    #true_beta_rho_s[4] = 0.2
    interaction_beta_x1_x2_rho_s = 0
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 1.0
    true_beta_rho_r[2] = 0.4
    #true_beta_rho_r[3] = 0.2
    #true_beta_rho_r[4] = 0.1
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.0
    true_beta_pi_r[1] = 1.1
    true_beta_pi_r[2] = 0.1
    #true_beta_pi_r[3] = 0.3
    #true_beta_pi_r[4] = 0.4
    interaction_beta_x2_x3_pi_r = -0.6

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s) + np.ravel(interaction_beta_x1_x2_rho_s*X["Covariate 1"]*(X["Covariate 2"].T)), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients_local,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r)  + np.ravel(interaction_beta_x2_x3_pi_r*X["Covariate 2"]*(X["Covariate 3"].T)), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3

def get_expected_theta_from_X_4_pi_rho(X): # Pi and rho_r
    N_patients_local, P = X.shape
    # These are the true parameters for a patient with all covariates equal to 0:
    #rho_s_population = -0.005
    #rho_r_population = 0.001
    #pi_r_population = 0.3
    rho_s_population = -0.04
    rho_r_population = 0.002
    pi_r_population = 0.1 #0.3
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    true_beta_rho_s = np.zeros(P)
    #true_beta_rho_s[0] = 0
    #true_beta_rho_s[1] = 0
    #true_beta_rho_s[2] = 0
    #true_beta_rho_s[3] = 0
    #true_beta_rho_s[4] = 0
    #interaction_beta_x1_x2_rho_r = 0
    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.4
    true_beta_rho_r[1] = 0
    true_beta_rho_r[2] = 0
    #true_beta_rho_r[3] = 0
    #true_beta_rho_r[4] = 0
    interaction_beta_x1_x2_rho_r = -0.8
    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0
    true_beta_pi_r[1] = 0.4
    true_beta_pi_r[2] = -0.6
    #true_beta_pi_r[3] = 0
    #true_beta_pi_r[4] = 0
    interaction_beta_x2_x3_pi_r = 1

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, true_beta_rho_s), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r) + np.ravel(interaction_beta_x1_x2_rho_r*X["Covariate 1"]*(X["Covariate 2"].T)), (N_patients_local,1))
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r)  + np.ravel(interaction_beta_x2_x3_pi_r*X["Covariate 2"]*(X["Covariate 3"].T)), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3
