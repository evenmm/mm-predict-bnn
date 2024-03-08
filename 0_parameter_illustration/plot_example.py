# -*- coding: utf-8 -*-
# 23.02.2024 Even Moa Myklebust

from utilities import *
import numpy as np
import matplotlib.pyplot as plt

max_time = 1 + 18*28
days_between_measurements = 28
measurement_times = np.array(range(1,max_time+days_between_measurements,days_between_measurements))
M_number_of_measurements = len(measurement_times)
treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

rho_s_population = -0.01
rho_r_population = 0.004
pi_r_population = 0.16
Y_0_population = 20

example_params = Parameters(Y_0=Y_0_population, pi_r=pi_r_population, g_r=rho_r_population, g_s=rho_s_population, k_1=0, sigma=0.5)
example_patient = Patient(example_params, measurement_times, treatment_history, name="")

name_arr = ["pi_r", "rho_s", "rho_r", "M"]

SAVEDIR = "./plots/"
def plot_example(patient, title, savename, PLOT_PARAMETERS=False, parameters = [], PLOT_lightchains=False, plot_pfs=False, plot_KapLam=False):
    measurement_times = patient.measurement_times
    if PLOT_PARAMETERS:
        plotting_times = np.linspace(measurement_times[0], measurement_times[-1], 80)
        treatment_history = np.array([Treatment(start=measurement_times[0], end=measurement_times[-1], id=1)])

        rho_s_increment = 0.003 #0.0015
        rho_r_increment = 0.0004 #0.0002
        pi_r_increment = 0.05 #0.015
        M_increment = 1.5 #5
        max_balfa = 2

        for maker in range(4):
            # pi_r
            if maker==0:
                fig, ax1 = plt.subplots()
                for balfa in range(1,max_balfa):
                    # More 
                    parameters.pi_r = pi_r_population + balfa*pi_r_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot sensitive M protein
                    ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    # Plot resistant M protein
                    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", alpha=0.4/balfa**2) #, label="True M protein (resistant)")
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")

                    # Less
                    parameters.pi_r = pi_r_population - balfa*pi_r_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot sensitive M protein
                    ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    # Plot resistant M protein
                    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", alpha=0.4/balfa**2) #, label="True M protein (resistant)")
                    parameters.pi_r = pi_r_population
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")
            
                # Plot true M protein curves according to parameters
                plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                # Count resistant part
                resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Plot sensitive M protein
                ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b') #, label="True M protein (total)")
                # Plot resistant M protein
                ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r") #, label="True M protein (resistant)")
                # Plot total M protein
                ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k') #, label="True M protein (total)")
                max_measurement_times = max(measurement_times) 
                #ax1.plot(patient.measurement_times, patient.Mprotein_values, "-", lw=1, alpha=0.5)
                ax1.set_title(title)
                ax1.set_xlabel("Days")
                ax1.set_ylabel("Serum M protein (g/L)")
                # xticks
                latest_cycle_start = int(np.rint((max_measurement_times-1)//28 + 1))
                tickresolution = 3
                tick_labels = [1] + [cs for cs in range(tickresolution, latest_cycle_start+1, tickresolution)]
                #tick_labels = [1] + [cs for cs in range(6, latest_cycle_start+1, 6)]
                tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
                ax1.set_xticks(tick_positions, tick_labels)
                ax1.set_xlabel("Cycle number")
                plt.tight_layout()
                plt.savefig(savename+"_"+name_arr[maker]+".pdf",dpi=300)
                #plt.show()
                plt.close()


            # rho_s
            if maker==1:
                fig, ax1 = plt.subplots()
                for balfa in range(1,max_balfa):
                    # More 
                    parameters.g_s = rho_s_population + balfa*rho_s_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot sensitive M protein
                    ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    
                    # Less
                    parameters.g_s = rho_s_population - balfa*rho_s_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot sensitive M protein
                    ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    parameters.g_s = rho_s_population        
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")

                # Plot true M protein curves according to parameters
                plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                # Count resistant part
                resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Plot sensitive M protein
                ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b') #, label="True M protein (total)")
                # Plot resistant M protein
                ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r") #, label="True M protein (resistant)")
                # Plot total M protein
                ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k') #, label="True M protein (total)")
                max_measurement_times = max(measurement_times) 
                #ax1.plot(patient.measurement_times, patient.Mprotein_values, "-", lw=1, alpha=0.5)
                ax1.set_title(title)
                ax1.set_xlabel("Days")
                ax1.set_ylabel("Serum M protein (g/L)")
                # xticks
                latest_cycle_start = int(np.rint((max_measurement_times-1)//28 + 1))
                tickresolution = 3
                tick_labels = [1] + [cs for cs in range(tickresolution, latest_cycle_start+1, tickresolution)]
                #tick_labels = [1] + [cs for cs in range(6, latest_cycle_start+1, 6)]
                tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
                ax1.set_xticks(tick_positions, tick_labels)
                ax1.set_xlabel("Cycle number")
                plt.tight_layout()
                plt.savefig(savename+"_"+name_arr[maker]+".pdf",dpi=300)
                #plt.show()
                plt.close()

            # rho_r
            if maker==2:
                fig, ax1 = plt.subplots()
                for balfa in range(1,max_balfa):
                    # More 
                    parameters.g_r = rho_r_population + balfa*rho_r_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot resistant M protein
                    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", alpha=0.4/balfa**2) #, label="True M protein (resistant)")
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")

                    # Less
                    parameters.g_r = rho_r_population - balfa*rho_r_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot resistant M protein
                    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", alpha=0.4/balfa**2)#, label="True M protein (resistant)")
                    parameters.g_r = rho_r_population
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")

                # Plot true M protein curves according to parameters
                plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                # Count resistant part
                resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Plot sensitive M protein
                ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b') #, label="True M protein (total)")
                # Plot resistant M protein
                ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r") #, label="True M protein (resistant)")
                # Plot total M protein
                ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k') #, label="True M protein (total)")
                max_measurement_times = max(measurement_times) 
                #ax1.plot(patient.measurement_times, patient.Mprotein_values, "-", lw=1, alpha=0.5)
                ax1.set_title(title)
                ax1.set_xlabel("Days")
                ax1.set_ylabel("Serum M protein (g/L)")
                # xticks
                latest_cycle_start = int(np.rint((max_measurement_times-1)//28 + 1))
                tickresolution = 3
                tick_labels = [1] + [cs for cs in range(tickresolution, latest_cycle_start+1, tickresolution)]
                #tick_labels = [1] + [cs for cs in range(6, latest_cycle_start+1, 6)]
                tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
                ax1.set_xticks(tick_positions, tick_labels)
                ax1.set_xlabel("Cycle number")
                plt.tight_layout()
                plt.savefig(savename+"_"+name_arr[maker]+".pdf",dpi=300)
                #plt.show()
                plt.close()

            # M
            if maker==3:
                fig, ax1 = plt.subplots()
                for balfa in range(1,max_balfa):
                    # More 
                    parameters.Y_0 = Y_0_population + balfa*M_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot sensitive M protein
                    ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    # Plot resistant M protein
                    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", alpha=0.4/balfa**2) #, label="True M protein (resistant)")
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")

                    # Less
                    parameters.Y_0 = Y_0_population - balfa*M_increment
                    # Plot true M protein curves according to parameters
                    plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                    # Count resistant part
                    resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                    plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                    # Plot sensitive M protein
                    ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    # Plot resistant M protein
                    ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", alpha=0.4/balfa**2) #, label="True M protein (resistant)")
                    # Plot total M protein
                    ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', alpha=0.4/balfa**2) #, label="True M protein (total)")
                    parameters.Y_0 = Y_0_population

                # Plot true M protein curves according to parameters
                plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
                # Count resistant part
                resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
                plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Plot sensitive M protein
                ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b') #, label="True M protein (total)")
                # Plot resistant M protein
                ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r") #, label="True M protein (resistant)")
                # Plot total M protein
                ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k') #, label="True M protein (total)")
                max_measurement_times = max(measurement_times) 
                #ax1.plot(patient.measurement_times, patient.Mprotein_values, "-", lw=1, alpha=0.5)
                ax1.set_title(title)
                ax1.set_xlabel("Days")
                ax1.set_ylabel("Serum M protein (g/L)")
                # xticks
                latest_cycle_start = int(np.rint((max_measurement_times-1)//28 + 1))
                tickresolution = 3
                tick_labels = [1] + [cs for cs in range(tickresolution, latest_cycle_start+1, tickresolution)]
                #tick_labels = [1] + [cs for cs in range(6, latest_cycle_start+1, 6)]
                tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
                ax1.set_xticks(tick_positions, tick_labels)
                ax1.set_xlabel("Cycle number")
                plt.tight_layout()
                plt.savefig(savename+"_"+name_arr[maker]+".pdf",dpi=300)
                #plt.show()
                plt.close()

plot_example(example_patient, "", SAVEDIR+"example_patient", PLOT_PARAMETERS=True, parameters=example_params, PLOT_lightchains=False, plot_pfs=False, plot_KapLam=False)
