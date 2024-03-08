from utilities import *
import arviz as az
import pymc as pm
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
##############################
# Function argument shapes: 
# X is an (N_patients, P) shaped pandas dataframe
# patient dictionary contains N_patients patients in the same order as X

def BNN_model(X, patient_dictionary, name, psi_prior="lognormal", MODEL_RANDOM_EFFECTS=True, FUNNEL_REPARAMETRIZATION=False, FUNNEL_WEIGHTS = False, WEIGHT_PRIOR = "symmetry_fix", SAVING=False, n_hidden = 3):
    df = pd.DataFrame(columns=["patient_id", "mprotein_value", "time"])
    for ii in range(len(patient_dictionary)):
        patient = patient_dictionary[ii]
        mprot = patient.Mprotein_values
        times = patient.measurement_times
        for jj in range(len(mprot)):
            single_entry = pd.DataFrame({"patient_id":[ii], "mprotein_value":[mprot[jj]], "time":[times[jj]]})
            df = pd.concat([df, single_entry], ignore_index=True)
    group_id = df["patient_id"].tolist()
    Y_flat_no_nans = np.array(df["mprotein_value"].tolist())
    t_flat_no_nans = np.array(df["time"].tolist())

    N_patients, P = X.shape
    P0 = int(P / 2) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    X_not_transformed = X.copy()
    X = X.T
    yi0 = np.zeros(N_patients)
    for ii in range(N_patients):
        yi0[ii] = patient_dictionary[ii].Mprotein_values[0]

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[0], n_hidden)
    if WEIGHT_PRIOR == "iso_normal":
        init_out = np.random.randn(n_hidden)
    else:
        init_out = abs(np.random.randn(n_hidden))

    with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as neural_net_model:
        # Observation noise (std)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

        # alpha
        alpha = pm.Normal("alpha",  mu=np.array([np.log(0.002), np.log(0.002), np.log(0.5/(1-0.5))]),  sigma=1, shape=3)

        log_sigma_weights_in = pm.Normal("log_sigma_weights_in", mu=2*np.log(0.01), sigma=2.5**2, shape=(X.shape[0], 1))
        sigma_weights_in = pm.Deterministic("sigma_weights_in", np.exp(log_sigma_weights_in))
        if FUNNEL_WEIGHTS == True: # Funnel reparametrized weights: 
            # Weights input to 1st layer
            weights_in_rho_s_offset = pm.Normal("weights_in_rho_s_offset ", mu=0, sigma=1, shape=(X.shape[0], n_hidden))
            weights_in_rho_r_offset = pm.Normal("weights_in_rho_r_offset ", mu=0, sigma=1, shape=(X.shape[0], n_hidden))
            weights_in_pi_r_offset = pm.Normal("weights_in_pi_r_offset ", mu=0, sigma=1, shape=(X.shape[0], n_hidden))
            weights_in_rho_s = pm.Deterministic("weights_in_rho_s", weights_in_rho_s_offset * np.repeat(sigma_weights_in, n_hidden, axis=1))
            weights_in_rho_r = pm.Deterministic("weights_in_rho_r", weights_in_rho_r_offset * np.repeat(sigma_weights_in, n_hidden, axis=1))
            weights_in_pi_r = pm.Deterministic("weights_in_pi_r", weights_in_pi_r_offset * np.repeat(sigma_weights_in, n_hidden, axis=1))
            # Weights from 1st to 2nd layer
            if WEIGHT_PRIOR == "iso_normal":
                weights_out_rho_s_offset = pm.Normal("weights_out_rho_s_offset ", mu=0, sigma=1, shape=(n_hidden, ))
                weights_out_rho_r_offset = pm.Normal("weights_out_rho_r_offset ", mu=0, sigma=1, shape=(n_hidden, ))
                weights_out_pi_r_offset = pm.Normal("weights_out_pi_r_offset ", mu=0, sigma=1, shape=(n_hidden, ))
            # WEIGHT_PRIOR == "Student_out" does not make sense with funnel
            else: # Handling symmetry
                weights_out_rho_s_offset = pm.HalfNormal("weights_out_rho_s_offset ", sigma=1, shape=(n_hidden, ))
                weights_out_rho_r_offset = pm.HalfNormal("weights_out_rho_r_offset ", sigma=1, shape=(n_hidden, ))
                weights_out_pi_r_offset = pm.HalfNormal("weights_out_pi_r_offset ", sigma=1, shape=(n_hidden, ))
            sigma_weights_out = pm.HalfNormal("sigma_weights_out", sigma=0.1)
            weights_out_rho_s = pm.Deterministic("weights_out_rho_s", weights_out_rho_s_offset * sigma_weights_out)
            weights_out_rho_r = pm.Deterministic("weights_out_rho_r", weights_out_rho_r_offset * sigma_weights_out)
            weights_out_pi_r = pm.Deterministic("weights_out_pi_r", weights_out_pi_r_offset * sigma_weights_out)
        else:
            # Weights input to 1st layer
            if WEIGHT_PRIOR == "Horseshoe":
                # Global shrinkage prior
                tau_in_rho_s = pm.HalfStudentT("tau_in_rho_s", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                tau_in_rho_r = pm.HalfStudentT("tau_in_rho_r", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                tau_in_pi_r = pm.HalfStudentT("tau_in_pi_r", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                # Local shrinkage prior
                lam_in_rho_s = pm.HalfStudentT("lam_in_rho_s", 2, shape=(X.shape[0], n_hidden)) #dims
                lam_in_rho_r = pm.HalfStudentT("lam_in_rho_r", 2, shape=(X.shape[0], n_hidden)) #dims
                lam_in_pi_r = pm.HalfStudentT("lam_in_pi_r", 2, shape=(X.shape[0], n_hidden)) #dims
                c2_in_rho_s = pm.InverseGamma("c2_in_rho_s", 1, 0.1)
                c2_in_rho_r = pm.InverseGamma("c2_in_rho_r", 1, 0.1)
                c2_in_pi_r = pm.InverseGamma("c2_in_pi_r", 1, 0.1)
                z_in_rho_s = pm.Normal("z_in_rho_s", 0.0, 1.0, shape=(X.shape[0], n_hidden)) #dims
                z_in_rho_r = pm.Normal("z_in_rho_r", 0.0, 1.0, shape=(X.shape[0], n_hidden)) #dims
                z_in_pi_r = pm.Normal("z_in_pi_r", 0.0, 1.0, shape=(X.shape[0], n_hidden)) #dims
                # Shrunken coefficients
                weights_in_rho_s = pm.Deterministic("weights_in_rho_s", z_in_rho_s * tau_in_rho_s * lam_in_rho_s * np.sqrt(c2_in_rho_s / (c2_in_rho_s + tau_in_rho_s**2 * lam_in_rho_s**2))) # dims
                weights_in_rho_r = pm.Deterministic("weights_in_rho_r", z_in_rho_r * tau_in_rho_r * lam_in_rho_r * np.sqrt(c2_in_rho_r / (c2_in_rho_r + tau_in_rho_r**2 * lam_in_rho_r**2))) # dims
                weights_in_pi_r = pm.Deterministic("weights_in_pi_r", z_in_pi_r * tau_in_pi_r * lam_in_pi_r * np.sqrt(c2_in_pi_r / (c2_in_pi_r + tau_in_pi_r**2 * lam_in_pi_r**2))) # dims
            else: 
                weights_in_rho_s = pm.Normal('weights_in_rho_s', 0, sigma=np.repeat(sigma_weights_in, n_hidden, axis=1), shape=(X.shape[0], n_hidden), initval=init_1)
                weights_in_rho_r = pm.Normal('weights_in_rho_r', 0, sigma=np.repeat(sigma_weights_in, n_hidden, axis=1), shape=(X.shape[0], n_hidden), initval=init_1)
                weights_in_pi_r = pm.Normal('weights_in_pi_r', 0, sigma=np.repeat(sigma_weights_in, n_hidden, axis=1), shape=(X.shape[0], n_hidden), initval=init_1)
            # Weights from 1st to 2nd layer
            if WEIGHT_PRIOR == "iso_normal":
                sigma_weights_out = pm.HalfNormal("sigma_weights_out", sigma=0.1)
                weights_out_rho_s = pm.Normal('weights_out_rho_s', 0, sigma=sigma_weights_out, shape=(n_hidden, ), initval=init_out)
                weights_out_rho_r = pm.Normal('weights_out_rho_r', 0, sigma=sigma_weights_out, shape=(n_hidden, ), initval=init_out)
                weights_out_pi_r = pm.Normal('weights_out_pi_r', 0, sigma=sigma_weights_out, shape=(n_hidden, ), initval=init_out)
            elif WEIGHT_PRIOR == "Student_out": # Handling symmetry
                weights_out_rho_s = pm.HalfStudentT('weights_out_rho_s', nu=4, sigma=1, shape=(n_hidden, ), initval=init_out)
                weights_out_rho_r = pm.HalfStudentT('weights_out_rho_r', nu=4, sigma=1, shape=(n_hidden, ), initval=init_out)
                weights_out_pi_r = pm.HalfStudentT('weights_out_pi_r', nu=4, sigma=1, shape=(n_hidden, ), initval=init_out)
            elif WEIGHT_PRIOR == "Horseshoe":
                # Global shrinkage prior
                tau_out_rho_s = pm.HalfStudentT("tau_out_rho_s", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                tau_out_rho_r = pm.HalfStudentT("tau_out_rho_r", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                tau_out_pi_r = pm.HalfStudentT("tau_out_pi_r", 2, P0 / (P - P0) * sigma_obs / np.sqrt(N_patients))
                # Local shrinkage prior
                lam_out_rho_s = pm.HalfStudentT("lam_out_rho_s", 2, shape=(n_hidden, )) #dims
                lam_out_rho_r = pm.HalfStudentT("lam_out_rho_r", 2, shape=(n_hidden, )) #dims
                lam_out_pi_r = pm.HalfStudentT("lam_out_pi_r", 2, shape=(n_hidden, )) #dims
                c2_out_rho_s = pm.InverseGamma("c2_out_rho_s", 1, 0.1)
                c2_out_rho_r = pm.InverseGamma("c2_out_rho_r", 1, 0.1)
                c2_out_pi_r = pm.InverseGamma("c2_out_pi_r", 1, 0.1)
                z_out_rho_s = pm.Normal("z_out_rho_s", 0.0, 1.0, shape=(n_hidden, )) #dims
                z_out_rho_r = pm.Normal("z_out_rho_r", 0.0, 1.0, shape=(n_hidden, )) #dims
                z_out_pi_r = pm.Normal("z_out_pi_r", 0.0, 1.0, shape=(n_hidden, )) #dims
                # Shrunken coefficients
                weights_out_rho_s = pm.Deterministic("weights_out_rho_s", z_out_rho_s * tau_out_rho_s * lam_out_rho_s * np.sqrt(c2_out_rho_s / (c2_out_rho_s + tau_out_rho_s**2 * lam_out_rho_s**2))) # dims
                weights_out_rho_r = pm.Deterministic("weights_out_rho_r", z_out_rho_r * tau_out_rho_r * lam_out_rho_r * np.sqrt(c2_out_rho_r / (c2_out_rho_r + tau_out_rho_r**2 * lam_out_rho_r**2))) # dims
                weights_out_pi_r = pm.Deterministic("weights_out_pi_r", z_out_pi_r * tau_out_pi_r * lam_out_pi_r * np.sqrt(c2_out_pi_r / (c2_out_pi_r + tau_out_pi_r**2 * lam_out_pi_r**2))) # dims
            else: # Handling symmetry
                sigma_weights_out = pm.HalfNormal("sigma_weights_out", sigma=0.1)
                weights_out_rho_s = pm.HalfNormal('weights_out_rho_s', sigma=sigma_weights_out, shape=(n_hidden, ), initval=init_out)
                weights_out_rho_r = pm.HalfNormal('weights_out_rho_r', sigma=sigma_weights_out, shape=(n_hidden, ), initval=init_out)
                weights_out_pi_r = pm.HalfNormal('weights_out_pi_r', sigma=sigma_weights_out, shape=(n_hidden, ), initval=init_out)

        # offsets for each node between each layer 
        sigma_bias_in = pm.HalfNormal("sigma_bias_in", sigma=1, shape=(1,n_hidden))
        bias_in_rho_s = pm.Normal("bias_in_rho_s", mu=0, sigma=sigma_bias_in, shape=(1,n_hidden)) # sigma=sigma_bias_in_rho_s
        bias_in_rho_r = pm.Normal("bias_in_rho_r", mu=0, sigma=sigma_bias_in, shape=(1,n_hidden)) # sigma=sigma_bias_in_rho_r
        bias_in_pi_r = pm.Normal("bias_in_pi_r", mu=0, sigma=sigma_bias_in, shape=(1,n_hidden)) # sigma=sigma_bias_in_pi_r
        
        # Calculate Y using neural net 
        # Leaky RELU activation
        pre_act_1_rho_s = pm.math.dot(X_not_transformed, weights_in_rho_s) + bias_in_rho_s
        pre_act_1_rho_r = pm.math.dot(X_not_transformed, weights_in_rho_r) + bias_in_rho_r
        pre_act_1_pi_r = pm.math.dot(X_not_transformed, weights_in_pi_r) + bias_in_pi_r
        act_1_rho_s = pm.math.switch(pre_act_1_rho_s > 0, pre_act_1_rho_s, pre_act_1_rho_s * 0.01)
        act_1_rho_r = pm.math.switch(pre_act_1_rho_r > 0, pre_act_1_rho_r, pre_act_1_rho_r * 0.01)
        act_1_pi_r = pm.math.switch(pre_act_1_pi_r > 0, pre_act_1_pi_r, pre_act_1_pi_r * 0.01)

        # Output activation function is just unit transform for prediction model
        # But then we put it through a sigmoid to avoid overflow (That made things worse)
        act_out_rho_s = pm.math.dot(act_1_rho_s, weights_out_rho_s)
        act_out_rho_r = pm.math.dot(act_1_rho_r, weights_out_rho_r)
        act_out_pi_r =  pm.math.dot(act_1_pi_r, weights_out_pi_r)

        # Latent variables theta
        omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
        if MODEL_RANDOM_EFFECTS:
            if FUNNEL_REPARAMETRIZATION == True: 
                # Reparametrized to escape/explore the funnel of Hell (https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/):
                theta_rho_s_offset = pm.Normal('theta_rho_s_offset', mu=0, sigma=1, shape=N_patients)
                theta_rho_r_offset = pm.Normal('theta_rho_r_offset', mu=0, sigma=1, shape=N_patients)
                theta_pi_r_offset  = pm.Normal('theta_pi_r_offset',  mu=0, sigma=1, shape=N_patients)
                theta_rho_s = pm.Deterministic("theta_rho_s", (alpha[0] + act_out_rho_s + theta_rho_s_offset * omega[0]))
                theta_rho_r = pm.Deterministic("theta_rho_r", (alpha[1] + act_out_rho_r + theta_rho_r_offset * omega[1]))
                theta_pi_r  = pm.Deterministic("theta_pi_r",  (alpha[2] + act_out_pi_r + theta_pi_r_offset  * omega[2]))
            else:
                # Original
                theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0] + act_out_rho_s, sigma=omega[0]) # Individual random intercepts in theta to confound effects of X
                theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + act_out_rho_r, sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
                theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + act_out_pi_r,  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X
        else: 
            theta_rho_s = pm.Deterministic("theta_rho_s", alpha[0] + act_out_rho_s)
            theta_rho_r = pm.Deterministic("theta_rho_r", alpha[1] + act_out_rho_r)
            theta_pi_r  = pm.Deterministic("theta_pi_r",  alpha[2] + act_out_pi_r)

        # psi: True M protein at time 0
        # 1) Normal. Fast convergence, but possibly negative tail 
        if psi_prior=="normal":
            psi = pm.Normal("psi", mu=yi0, sigma=sigma_obs, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        # 2) Lognormal. Works if you give it time to converge
        if psi_prior=="lognormal":
            xi = pm.HalfNormal("xi", sigma=1)
            log_psi = pm.Normal("log_psi", mu=np.log(yi0+1e-8), sigma=xi, shape=N_patients)
            psi = pm.Deterministic("psi", np.exp(log_psi))

        # Transformed latent variables 
        rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
        rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
        pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

        # Observation model 
        mu_Y = psi[group_id] * (pi_r[group_id]*np.exp(rho_r[group_id]*t_flat_no_nans) + (1-pi_r[group_id])*np.exp(rho_s[group_id]*t_flat_no_nans))

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y_flat_no_nans)
    # Visualize model
    #import graphviz 
    #gv = pm.model_to_graphviz(neural_net_model) # With shared vcariables: --> 170 assert force_compile or (version == get_version())   AssertionError.
    #gv.render(filename="./plots/posterior_plots/"+name+"_graph_of_model", format="png", view=False)
    # Sample from prior:
    """
    with neural_net_model:
        prior_samples = pm.sample_prior_predictive(200)
    raveled_Y_sample = np.ravel(prior_samples.prior_predictive["Y_obs"])
    # Below plotlimit_prior
    plotlimit_prior = 1000
    plt.figure()
    az.plot_dist(Y_flat_no_nans[Y_flat_no_nans<plotlimit_prior], color="C1", label="observed", bw=3)
    az.plot_dist(raveled_Y_sample[raveled_Y_sample<plotlimit_prior], label="simulated", bw=3)
    plt.title("Samples from prior compared to observations, for Y<plotlimit_prior")
    plt.xlabel("Y (M protein)")
    plt.ylabel("Frequency")
    if SAVING:
        plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples_below_"+str(plotlimit_prior)+".png")
    plt.close()
    # All samples: 
    plt.figure()
    az.plot_dist(Y_flat_no_nans, color="C1", label="observed", bw=3)
    az.plot_dist(raveled_Y_sample, label="simulated", bw=3)
    plt.title("Samples from prior compared to observations")
    plt.xlabel("Y (M protein)")
    plt.ylabel("Frequency")
    if SAVING:
        plt.savefig("./plots/posterior_plots/"+name+"-plot_prior_samples.png")
    plt.close()
    """
    return neural_net_model
