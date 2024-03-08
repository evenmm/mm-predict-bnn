from utilities import * 
from multiprocessing import Pool

def get_posterior_median(args): # Predicts observations of M protein
    # plotting_times as input because it is common between patients 
    # In this function we predict the M protein values for a patient using the posterior samples of the parameters
    # We also calculate the mean squared error of the predicted values with respect to the observed values
    # The errors are returned for plotting in function plot_error_of_mean_in_simdata
    sample_shape, M_number_of_measurements, ii, idata, X_test, patient_dictionary_test, SAVEDIR, name, N_rand_eff_pred, N_rand_obs_pred, model_name, parameter_dictionary_test, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, PLOT_RESISTANT = args
    if not CI_with_obs_noise:
        N_rand_eff_pred = N_rand_eff_pred * N_rand_obs_pred
        N_rand_obs_pred = 1
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    np.random.seed(ii) # Seeding the randomness in observation noise sigma, in random effects and in psi = yi0 + random(sigma)
    #patient = patient_dictionary_test["Patient " + str(ii)]
    patient = patient_dictionary_test[ii]
    print(patient.name) #'Name of the patient is 0, but label in patient_dict is "Patient 0"'
    measurement_times = patient.get_measurement_times() 
    treatment_history = patient.get_treatment_history()
    pad_d = M_number_of_measurements - len(measurement_times)
    plotting_times = np.pad(measurement_times, (0, pad_d), 'constant', constant_values=measurement_times[-1]) #measurement_times #np.linspace(first_time, max_time, M_number_of_measurements) #int((measurement_times[-1]+1)*10))
    predicted_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
    predicted_y_values = np.empty(shape=(n_chains*N_rand_eff_pred, n_samples*N_rand_obs_pred, len(plotting_times)))
    predicted_y_resistant_values = np.empty_like(predicted_y_values)
    for ch in range(n_chains):
        for sa in range(n_samples):
            sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
            alpha = np.ravel(idata.posterior['alpha'][ch,sa])

            if model_name == "linear": 
                #this_beta_rho_s = np.ravel(idata.posterior['beta_rho_s'][ch,sa])
                this_beta_rho_r = np.ravel(idata.posterior['beta_rho_r'][ch,sa])
                this_beta_pi_r = np.ravel(idata.posterior['beta_pi_r'][ch,sa])
            elif model_name == "BNN": 
                #if "rho_s" in net_list:
                weights_in_rho_s = idata.posterior['weights_in_rho_s'][ch,sa]
                weights_out_rho_s = idata.posterior['weights_out_rho_s'][ch,sa]
                bias_in_rho_s = np.ravel(idata.posterior['bias_in_rho_s'][ch,sa])
                pre_act_1_rho_s = np.dot(X_test.iloc[ii,:], weights_in_rho_s) + bias_in_rho_s
                act_1_rho_s = np.select([pre_act_1_rho_s > 0, pre_act_1_rho_s <= 0], [pre_act_1_rho_s, pre_act_1_rho_s*0.01], 0)
                act_out_rho_s = np.dot(act_1_rho_s, weights_out_rho_s)
                #else: 
                #    act_out_rho_s = 0

                #if "rho_r" in net_list:
                weights_in_rho_r = idata.posterior['weights_in_rho_r'][ch,sa]
                weights_out_rho_r = idata.posterior['weights_out_rho_r'][ch,sa]
                bias_in_rho_r = np.ravel(idata.posterior['bias_in_rho_r'][ch,sa])
                pre_act_1_rho_r = np.dot(X_test.iloc[ii,:], weights_in_rho_r) + bias_in_rho_r
                act_1_rho_r = np.select([pre_act_1_rho_r > 0, pre_act_1_rho_r <= 0], [pre_act_1_rho_r, pre_act_1_rho_r*0.01], 0)
                act_out_rho_r = np.dot(act_1_rho_r, weights_out_rho_r)
                #else:
                #    act_out_rho_r = 0

                #if "pi_r" in net_list:
                weights_in_pi_r = idata.posterior['weights_in_pi_r'][ch,sa]
                weights_out_pi_r = idata.posterior['weights_out_pi_r'][ch,sa]
                bias_in_pi_r = np.ravel(idata.posterior['bias_in_pi_r'][ch,sa])
                pre_act_1_pi_r  = np.dot(X_test.iloc[ii,:], weights_in_pi_r)  + bias_in_pi_r
                act_1_pi_r =  np.select([pre_act_1_pi_r  > 0, pre_act_1_pi_r  <= 0], [pre_act_1_pi_r,  pre_act_1_pi_r*0.01],  0)
                act_out_pi_r =  np.dot(act_1_pi_r,  weights_out_pi_r)
                #else:
                #    act_out_pi_r = 0

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
                    #predicted_theta_1 = np.random.normal(alpha[0] + np.dot(X_test.iloc[ii,:], this_beta_rho_s), omega[0])
                    predicted_theta_1 = np.random.normal(alpha[0], omega[0])
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
                        noise_array = np.random.normal(0, sigma_obs, M_number_of_measurements)
                        noisy_observations = predicted_y_values_noiseless + noise_array
                        predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                        predicted_y_resistant_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] = predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
                else: 
                    predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa] = predicted_y_values_noiseless
                    predicted_y_resistant_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa] = predicted_y_values[N_rand_eff_pred*ch + ee, N_rand_obs_pred*sa] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
    flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*n_samples*N_rand_eff_pred*N_rand_obs_pred,M_number_of_measurements))
    sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
    print(len(sorted_local_pred_y_values))
    #upper_limits = sorted_pred_y_values[upper_index,training_instance_id,:]       #color=color_array[index]
    posterior_median = np.median(sorted_local_pred_y_values, axis=0)
    print("posterior_median", posterior_median)
    #return posterior_median # {"posterior_parameters" : posterior_parameters, "predicted_y_values" : predicted_y_values, "predicted_y_resistant_values" : predicted_y_resistant_values}
    pad_n = M_number_of_measurements - len(posterior_median)
    return np.pad(posterior_median, (0, pad_n), 'constant', constant_values=np.nan)
