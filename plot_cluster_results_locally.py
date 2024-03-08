from utilities import *
from BNN_model import *
from linear_model import *
from simdata_support import *

# Initialize random number generator
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
print(f"Running on PyMC v{pm.__version__}")
PLOT_SAVEDIR = "./plots/" # "/data/evenmm/plots/"
PICKLE_DIR = "./binaries_and_pickles/" #"/data/evenmm/binaries_and_pickles/"

#script_index = int(sys.argv[1]) 
for script_index in [1,2,4,5]:
    PLOT_INDIVIDUAL_PATIENTS = True

    # Settings
    if int(script_index % 3) == 0:
        true_sigma_obs = 0
    elif int(script_index % 3) == 1:
        true_sigma_obs = 1
    elif int(script_index % 3) == 2:
        true_sigma_obs = 2.5

    if script_index >= 3:
        RANDOM_EFFECTS = True
    else: 
        RANDOM_EFFECTS = False

    RANDOM_EFFECTS_TEST = False

    # These can be tampered with, remember to check all
    N_patients = 150
    N_patients_test = 150
    advi_iterations_BNN = 100_000
    advi_iterations_lin = 100_000 # 600_000
    N_samples = 10_000
    N_tuning = 10_000

    n_chains = 4
    psi_prior="lognormal"
    WEIGHT_PRIOR = "Student_out" #"Horseshoe" # "Student_out" #"symmetry_fix" #"iso_normal" "Student_out"
    ADADELTA_BNN = True
    ADADELTA_lin = True #False
    target_accept = 0.99
    CI_with_obs_noise = True
    PLOT_RESISTANT = True
    FUNNEL_REPARAMETRIZATION = False
    MODEL_RANDOM_EFFECTS = True
    DIFFERENT_LENGTHS = True
    min_length = 5
    #crop_after_pfs = False # not implemented
    N_HIDDEN = 2
    P = 5 # Number of covariates
    P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    true_omega = np.array([0.10, 0.05, 0.20])
    theta_generator = get_expected_theta_from_X_4_pi_rho #get_expected_theta_from_X_2
    test_seed = 23

    M_number_of_measurements = 14
    y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
    true_omega_for_psi = 0.1

    max_time = 28*M_number_of_measurements #400 #1200 #3000 #1500
    days_between_measurements = int(max_time/M_number_of_measurements)
    measurement_times = days_between_measurements * np.linspace(0, M_number_of_measurements, M_number_of_measurements)
    treatment_history = np.array([Treatment(start=0, end=measurement_times[-1], id=1)])

    # Generate train and test patients
    X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r = generate_simulated_patients(deepcopy(measurement_times), treatment_history, true_sigma_obs, N_patients, P, theta_generator, true_omega, true_omega_for_psi, seed=42, RANDOM_EFFECTS=RANDOM_EFFECTS, DIFFERENT_LENGTHS=DIFFERENT_LENGTHS, min_length=min_length)

    # Generate test patients
    X_test, patient_dictionary_test, parameter_dictionary_test, expected_theta_1_test, true_theta_rho_s_test, true_rho_s_test, expected_theta_2_test, true_theta_rho_r_test, true_rho_r_test, expected_theta_3_test, true_theta_pi_r_test, true_pi_r_test = generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_test, P, theta_generator, true_omega, true_omega_for_psi, seed=test_seed, RANDOM_EFFECTS=RANDOM_EFFECTS_TEST, DIFFERENT_LENGTHS=DIFFERENT_LENGTHS, min_length=min_length)
    print("Done generating train and test patients")

    # Visualize parameter dependancy on covariates 
    model_name_BNN = "BNN"
    name_BNN = "simdata_"+model_name_BNN+"_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_FUNNEL_"+str(FUNNEL_REPARAMETRIZATION)+"_RNDM_EFFECTS_"+str(RANDOM_EFFECTS)+"_WT_PRIOR_"+str(WEIGHT_PRIOR+"_N_HIDDN_"+str(N_HIDDEN))
    #plot_parameter_dependency_on_covariates(PLOT_SAVEDIR, name_BNN, X, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r)

    # plot all M protein values of all train patients together
    plt.figure()
    for patient_name, patient in patient_dictionary.items():
        plt.plot(patient.measurement_times, patient.Mprotein_values, linestyle="-", marker=".", linewidth=1) #, color="black")
    plt.ylabel("Serum Mprotein (g/L)")
    plt.xlabel("Time (days)")
    plt.savefig(PLOT_SAVEDIR+"all_measurements_"+name_BNN+".pdf", dpi=300)
    #plt.show()
    plt.close()

    # Sample from full BNN model
    neural_net_model = BNN_model(X, patient_dictionary, name_BNN, psi_prior=psi_prior, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, WEIGHT_PRIOR=WEIGHT_PRIOR, n_hidden=N_HIDDEN)
    # Draw samples from posterior:
    try: 
        # Load idata
        print("Attempting to load "+name_BNN)
        picklefile = open(PICKLE_DIR+name_BNN+'_idata', 'rb')
        idata_BNN = pickle.load(picklefile)
        picklefile.close()
        time.sleep(1)
        print("Loaded idata BNN")
    except:
        print("Running "+name_BNN)
        picklefile = open(PICKLE_DIR+name_BNN+'_idata', 'wb')
        with neural_net_model:
            if ADADELTA_BNN: 
                print("------------------- ADADELTA INITIALIZATION -------------------")
                advi = pm.ADVI()
                tracker = pm.callbacks.Tracker(
                    mean=advi.approx.mean.eval,  # callable that returns mean
                    std=advi.approx.std.eval,  # callable that returns std
                )
                approx = advi.fit(advi_iterations_BNN, obj_optimizer=pm.adadelta(), obj_n_mc=50, callbacks=[tracker])
                #approx = advi.fit(advi_iterations_BNN, obj_optimizer=pm.adagrad(), obj_n_mc=5, callbacks=[tracker])

                # Plot ELBO and trace
                fig = plt.figure(figsize=(16, 9))
                mu_ax = fig.add_subplot(221)
                std_ax = fig.add_subplot(222)
                hist_ax = fig.add_subplot(212)
                mu_ax.plot(tracker["mean"])
                mu_ax.set_title("Mean track")
                std_ax.plot(tracker["std"])
                std_ax.set_title("Std track")
                hist_ax.plot(advi.hist)
                hist_ax.set_title("Negative ELBO track")
                hist_ax.set_yscale("log")
                plt.savefig(PLOT_SAVEDIR+"0_elbo_and_trace_"+name_BNN+".pdf", dpi=300)
                #plt.show()
                plt.close()
                
                print("-------------------SAMPLING-------------------")
                # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
                scaling = approx.cov.eval()
                sample = approx.sample(return_inferencedata=False, size=n_chains)
                start_dict = list(sample[i] for i in range(n_chains))    
                # essentially, this is what init='advi' does!!!
                step = pm.NUTS(scaling=scaling, is_cov=True)
                idata_BNN = pm.sample(draws=N_samples, tune=N_tuning, step=step, chains=n_chains, initvals=start_dict) #, random_seed=42, target_accept=target_accept)
            else: 
                idata_BNN = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=60000, random_seed=42, target_accept=target_accept)
        print("Done sampling BNN model")
        pickle.dump(idata_BNN, picklefile)
        picklefile.close()

    """
    quasi_geweke_test(idata_BNN, model_name=model_name_BNN, first=0.1, last=0.5)
    plot_posterior_traces(idata_BNN, PLOT_SAVEDIR, name_BNN, psi_prior, model_name=model_name_BNN)
    """
    if PLOT_INDIVIDUAL_PATIENTS:
        plot_all_credible_intervals(idata_BNN, patient_dictionary, patient_dictionary_test, X_test, PLOT_SAVEDIR, name_BNN, y_resolution, model_name=model_name_BNN, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=True)
    print("BNN plots finished!")


    ### Linear model 
    # Sample from full model
    model_name_linear = "linear"
    name_lin = "simdata_"+model_name_linear+"_"+str(script_index)+"_M_"+str(M_number_of_measurements)+"_P_"+str(P)+"_N_pax_"+str(N_patients)+"_N_sampl_"+str(N_samples)+"_N_tune_"+str(N_tuning)+"_FUNNEL_"+str(FUNNEL_REPARAMETRIZATION)+"_RNDM_EFFECTS_"+str(RANDOM_EFFECTS)
    lin_model = linear_model(X, patient_dictionary, name_lin, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION)
    # Draw samples from posterior:
    try: 
        # Load idata
        print("Attempting to load "+name_lin)
        picklefile = open(PICKLE_DIR+name_lin+'_idata', 'rb')
        idata_lin = pickle.load(picklefile)
        picklefile.close()
        time.sleep(1)
        print("Loaded idata lin")
    except:
        print("Running "+name_lin)
        picklefile = open(PICKLE_DIR+name_lin+'_idata', 'wb')
        with lin_model:
            if ADADELTA_lin:
                print("------------------- INDEPENDENT ADVI -------------------")
                advi = pm.ADVI()
                tracker = pm.callbacks.Tracker(
                    mean=advi.approx.mean.eval,  # callable that returns mean
                    std=advi.approx.std.eval,  # callable that returns std
                )
                approx = advi.fit(advi_iterations_lin, obj_optimizer=pm.adadelta(), callbacks=[tracker])
                # Plot ELBO and trace
                fig = plt.figure(figsize=(16, 9))
                mu_ax = fig.add_subplot(221)
                std_ax = fig.add_subplot(222)
                hist_ax = fig.add_subplot(212)
                mu_ax.plot(tracker["mean"])
                mu_ax.set_title("Mean track")
                std_ax.plot(tracker["std"])
                std_ax.set_title("Std track")
                hist_ax.plot(advi.hist)
                hist_ax.set_title("Negative ELBO track")
                hist_ax.set_yscale("log")
                plt.savefig(PLOT_SAVEDIR+"0_elbo_and_trace_"+name_BNN+".pdf", dpi=300)
                #plt.show()
                plt.close()
                print("-------------------SAMPLING-------------------")
                # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
                scaling = approx.cov.eval()
                sample = approx.sample(return_inferencedata=False, size=n_chains)
                start_dict = list(sample[i] for i in range(n_chains))    
                # essentially, this is what init='advi' does!!!
                step = pm.NUTS(scaling=scaling, is_cov=True)
                idata_lin = pm.sample(draws=N_samples, tune=N_tuning, step=step, chains=n_chains, initvals=start_dict) #, random_seed=42, target_accept=target_accept)
            else:
                idata_lin = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=600, random_seed=42, target_accept=target_accept, chains=n_chains, cores=1)
        print("Done sampling linearmodel")
        pickle.dump(idata_lin, picklefile)
        picklefile.close()

    """
    quasi_geweke_test(idata_lin, model_name=model_name_linear, first=0.1, last=0.5)
    plot_posterior_traces(idata_lin, PLOT_SAVEDIR, name_lin, psi_prior, model_name=model_name_linear)
    """
    if PLOT_INDIVIDUAL_PATIENTS:
        plot_all_credible_intervals(idata_lin, patient_dictionary, patient_dictionary_test, X_test, PLOT_SAVEDIR, name_lin, y_resolution, model_name=model_name_linear, parameter_dictionary=parameter_dictionary, PLOT_PARAMETERS=True, parameter_dictionary_test=parameter_dictionary_test, PLOT_PARAMETERS_test=True, PLOT_TREATMENTS=False, MODEL_RANDOM_EFFECTS=MODEL_RANDOM_EFFECTS, CI_with_obs_noise=CI_with_obs_noise, PLOT_RESISTANT=True, PARALLELLIZE=False)
    print("Linear model plots finished!")

    def get_absolute_errors_and_posterior_medians(local_name, PLOT_SAVEDIR, local_idata, local_model_name):
        sample_shape = local_idata.posterior['psi'].shape # [chain, N_samples, dim]
        N_chains = sample_shape[0]
        N_samples = sample_shape[1]
        var_dimensions = sample_shape[2] # one per patient

        # Posterior predictive CI for test data
        if N_samples <= 10:
            N_rand_eff_pred = 100 # Number of random intercept samples to draw for each local_idata sample when we make predictions
            N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
        elif N_samples <= 100:
            N_rand_eff_pred = 10 # Number of random intercept samples to draw for each local_idata sample when we make predictions
            N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
        elif N_samples <= 1000:
            N_rand_eff_pred = 1 # Number of random intercept samples to draw for each local_idata sample when we make predictions
            N_rand_obs_pred = 100 # Number of observation noise samples to draw for each parameter sample 
        else:
            N_rand_eff_pred = 1 # Number of random intercept samples to draw for each local_idata sample when we make predictions
            N_rand_obs_pred = 10 # Number of observation noise samples to draw for each parameter sample 
        print("Number of samples set in plot_error_of_mean_in_simdata")

        #ax.errorbar(x=x_locations+offset, y=meanvals, yerr=critical_value_with_correct_dof*(np.sqrt(sum_of_square_error_values/degrees_of_freedom))/np.sqrt(n_splits), fmt=".", color=bar_color_array[model_index]) #, capsize=1, markersize=0, elinewidth=1) #fmt=".", color=color_array[model_index]) # markersize=5, linewidth=1
        N_patients_test = len(patient_dictionary_test)
        args = [(sample_shape, M_number_of_measurements, ii, local_idata, X_test, patient_dictionary_test, PLOT_SAVEDIR, local_name, N_rand_eff_pred, N_rand_obs_pred, local_model_name, parameter_dictionary_test, MODEL_RANDOM_EFFECTS, CI_with_obs_noise, PLOT_RESISTANT) for ii in range(N_patients_test)]
        try:
            picklefile = open(PICKLE_DIR+local_name+'all_posterior_medians', "rb")
            all_posterior_medians = pickle.load(picklefile)
            picklefile.close()
        except: 
            PARALLELLIZE = True
            if PARALLELLIZE:
                print("Running pool...")
                with Pool(15) as pool:
                    all_posterior_medians = pool.map(get_posterior_median,args)
            else: 
                all_posterior_medians = np.zeros((N_patients_test, M_number_of_measurements))
                for patient in patient_dictionary_test.values():
                    print(patient.local_name)
                    
                for ll, elem in enumerate(args):
                    all_posterior_medians[ll,:] = get_posterior_median(elem)
            print("...done.")
            print(all_posterior_medians)
            picklefile = open(PICKLE_DIR+local_name+'all_posterior_medians', "wb")
            pickle.dump(all_posterior_medians, picklefile)
            picklefile.close()

        # get intervals for all_posterior_medians
        plt.figure()
        for elem in all_posterior_medians:
            plt.plot(range(len(elem)), elem)
        #plt.show()
        plt.close()

        all_errors = np.zeros((len(all_posterior_medians), len(all_posterior_medians[0])))
        for ii, elem in enumerate(all_posterior_medians):
            #all_errors[ii,:] = patient_dictionary_test["Patient "+str(ii)].Mprotein_values - elem
            mprot = patient_dictionary_test[ii].Mprotein_values
            errors_unpad = mprot - elem[0:len(mprot)]
            all_errors[ii,:] = np.pad(errors_unpad, (0, len(elem)-len(mprot)), 'constant', constant_values=np.nan)

        all_absolute_errors = np.abs(all_errors)
        all_errors = None

        all_absolute_errors_sorted = np.sort(all_absolute_errors, axis=0)
        print("all_absolute_errors_sorted:\n", all_absolute_errors_sorted)
        return all_absolute_errors_sorted, all_posterior_medians

    all_absolute_errors_sorted_BNN, all_posterior_medians_BNN = get_absolute_errors_and_posterior_medians(name_BNN, PLOT_SAVEDIR, idata_BNN, "BNN") #, patient_dictionary_test, M_number_of_measurements)
    all_absolute_errors_sorted_lin, all_posterior_medians_lin = get_absolute_errors_and_posterior_medians(name_lin, PLOT_SAVEDIR, idata_lin, "linear") #, patient_dictionary_test, M_number_of_measurements)

    plt.figure()
    for elem in all_absolute_errors_sorted_BNN:
        plt.plot(range(len(elem)), elem)
    plt.title("BNN")
    #plt.savefig(PLOT_SAVEDIR+"all_absolute_errors_sorted_BNN_"+name_BNN+".pdf", dpi=300)

    plt.figure()
    for elem in all_absolute_errors_sorted_lin:
        plt.plot(range(len(elem)), elem)
    plt.title("Linear")
    #plt.savefig(PLOT_SAVEDIR+"all_absolute_errors_sorted_lin_"+name_BNN+".pdf", dpi=300)
    #plt.show()
    plt.close()

    # Filter out nans 
    #print(all_absolute_errors_sorted_BNN)
    #all_absolute_errors_sorted_BNN_no_nan = all_absolute_errors_sorted_BNN[~np.isnan(all_absolute_errors_sorted_BNN)]
    #all_absolute_errors_sorted_lin_no_nan = all_absolute_errors_sorted_lin[~np.isnan(all_absolute_errors_sorted_lin)]

    # Filter data using np.isnan
    def filter_nan(data):
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
        return filtered_data # a list!

    # Apply it to all_absolute errors lin and BNN
    all_absolute_errors_sorted_BNN_no_nan = filter_nan(all_absolute_errors_sorted_BNN)
    all_absolute_errors_sorted_lin_no_nan = filter_nan(all_absolute_errors_sorted_lin)
    print("all_absolute_errors_sorted_BNN_no_nan", all_absolute_errors_sorted_BNN_no_nan)
    print("all_absolute_errors_sorted_lin_no_nan", all_absolute_errors_sorted_lin_no_nan)

    """
    all_absolute_errors_sorted_BNN_no_nan = [elem[~np.isnan(elem)] for elem in all_absolute_errors_sorted_BNN]# all_absolute_errors_sorted_BNN[~np.isnan(all_absolute_errors_sorted_BNN)]
    all_absolute_errors_sorted_lin_no_nan = [elem[~np.isnan(elem)] for elem in all_absolute_errors_sorted_lin]# all_absolute_errors_sorted_lin[~np.isnan(all_absolute_errors_sorted_lin)]
    print(all_absolute_errors_sorted_BNN_no_nan)
    """

    # [1:] is to remove time zero, which is given so error 0

    # Boxplot with labels
    fig, ax = plt.subplots()
    plotting_times = np.array(range(1, M_number_of_measurements))
    ax.boxplot(all_absolute_errors_sorted_BNN_no_nan[1:], widths=0.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor="lightgrey"))
    #plt.savefig(PLOT_SAVEDIR+"BNN_abs_errors_xaxis_"+name_BNN+".pdf", dpi=300)
    #plt.show()

    # Boxplot with labels
    fig, ax = plt.subplots()
    ax.boxplot(all_absolute_errors_sorted_BNN_no_nan[1:], positions=plotting_times+0.2, widths=0.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor="lightblue")) #, boxprops=dict(facecolor="lightgrey"))
    ax.boxplot(all_absolute_errors_sorted_lin_no_nan[1:], positions=plotting_times-0.2, widths=0.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor="lightgrey")) #, boxprops=dict(facecolor="lightblue"))
    # Plot median too in the same color
    #ax.plot(plotting_times+0.2, np.median(all_absolute_errors_sorted_BNN_no_nan, axis=0), label="BNN model", color="orange", marker="None", linestyle="None")
    #ax.plot(plotting_times-0.2, np.median(all_absolute_errors_sorted_lin_no_nan, axis=0), label="Linear model", color="blue", marker="None", linestyle="None")

    # create patch for legend
    import matplotlib.patches as mpatches
    patch1 = mpatches.Patch(color="lightgrey", label="Linear model")
    patch2= mpatches.Patch(color="lightblue", label="BNN model")
    plt.legend(handles=[patch1, patch2])

    # Set the x-axis ticks to integer values
    min_time = int(np.floor(min(plotting_times)))
    max_time = int(np.ceil(max(plotting_times)))
    ticks = range(min_time, max_time + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlabel("Cycles after treatment start")
    ax.set_ylabel("Absolute error in M protein value")
    ax.set_title("Absolute errors in M protein value")
    plt.savefig(PLOT_SAVEDIR+"BNN_abs_errors_"+name_BNN+".pdf", dpi=300)
    #plt.show()

