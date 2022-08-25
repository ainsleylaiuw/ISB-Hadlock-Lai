# under TODO
# implemented 8/22:
#   model subsetting for derivatives on LHS
#   weights for thresholds. Need to check formatting
inputs_temp = np.tile([range(nPCs)], 3)
inputs_per_library = np.reshape(inputs_temp, (3, nPCs))
Sym_eqns = []
# need to make code that will determine model LHS (from features) 
# that contain 'dot' in feature name and get their model index to use in model_subset
for i in range(nPCs):
    inputs_per_library[2, :] = i
    print(inputs_per_library)
    #print(inputs_per_library)
    #print()
    generalized_library = ps.GeneralizedLibrary(
        [constant_library, custom_library, sindy_library],
        tensor_array=[[0,1,1]],
        inputs_per_library=inputs_per_library
    )
    feature_list = generalized_library.get_feature_names()
    # takes index of functions w/ derivative
    # should be same indexes for all PC but need to check later when graphing/simulating
    
    # need to convert to array
    model_subset = [i for i, j in enumerate(feature_list) if i.__contains__('dot')
    weights = np.ones(len(feature_list)) * 0.5
    # promotes derivative terms
    # to be used with all models
    for i in range(len(model_subset)):
        weights[model_subset[i]] = 0.1
    weights = np.tile([weights], nPCs) # not sure if [] is correct format
    sindy_opt = ps.SINDyPI(
        threshold=1e-1,
        tol=1e-8,
        thresholder="l1",
        max_iter=40000,
        model_subset=model_subset,
        #thresholds=weights
    )
    model = ps.SINDy(feature_library=generalized_library,
                     optimizer=sindy_opt,
                     differentiation_method=ps.FiniteDifference(drop_endpoints=False),
                    )
    model.fit(x_train[0], t=tvals, multiple_trajectories=False)
    #model.fit(x_train, t=tvals, multiple_trajectories=True)
    print(model.get_feature_names())
    #returns [sym_equations_simplified, sym_equations_rounded_simplified]
    sym_eqn = format_eqn(model, PC_index=i, r=nPCs) 
    Sym_eqns.append(sym_eqn)
    
#####################################################################################

# edits to be made to def format_eqn()

def format_eqn(model, PC_index, r): 
    features = model.get_feature_names()
    features_copy = list(np.copy(features))
    nfeatures = len(features)
    features_formatted = []
    """# for use with poly_library
    for i in range(nfeatures):
        temp_string = features[i].replace(" ", "*")
        features[i] = temp_string
    """
    # Need to put multiplication between terms for sympy
    for i in range(nfeatures):
        for j in range(r):
            # Overkill to make sure all the x0, x1, etc. get replaced
            temp_string = features[i].replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + " x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + " x", "x" + str(j) + " * x")
            features[i] = temp_string
        features_formatted.append(temp_string)
    features = features_copy

    coefs = model.coefficients()
    # should "clean" NaNs and replace w/ 0
    # consider moving it and coefs = model... up
    # https://www.codespeedy.com/check-if-a-numpy-array-contains-any-nan-value-in-python/
    for i in range(nfeatures):
        temp_string = features[i].replace(" ", "")
        features[i] = temp_string
        x = np.isnan(coefs[i])
        coefs[i][x] = 0
    sym_features = [sp.symbols(feature) for feature in features]
    sym_theta = [sp.symbols(feature) for feature in features]
    #print(sym_theta)

    sym_equations = []
    sym_equations_rounded = []
    for i in range(nfeatures):
        sym_equations.append(
            sp.solve(
                sp.Eq(sym_theta[i], sym_theta @ np.around(coefs[i], 10)), sym_features[i]
            )
        )
        sym_equations_rounded.append(
            sp.solve(
                sp.Eq(sym_theta[i], sym_theta @ np.around(coefs[i], 2)), sym_features[i]
            )
        )
        #print(sym_theta[i], " = ", sym_equations_rounded[i][0])

    # Define the ODE symbol variables
    t_sym = sp.symbols("t_sym")
    x_sym = sp.symbols("x:%d" % r)
    x_dot_sym = sp.symbols("x:%d_dot" % r)
    #print(x_dot_sym)
    # Need to format the above equations so that there are space between x0 * x0, x0 * x_dot0, and so on.
    sym_equations_formatted = []
    sym_equations_rounded_formatted = []
    for i in range(nfeatures):
        temp_string = str(sym_equations[i])
        temp_rounded_string = str(sym_equations_rounded[i])
        for j in range(r):
            # Overkill to make sure all the x0, x1, etc. get replaced
            temp_string = temp_string.replace(
                "x" + str(j) + "x", "x" + str(j) + " * x"
            )
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")
            temp_string = temp_string.replace("x" + str(j) + "x", "x" + str(j) + " * x")

            temp_rounded_string = temp_rounded_string.replace(
                "x" + str(j) + "x", "x" + str(j) + " * x"
            )
            temp_rounded_string = temp_rounded_string.replace(
                "x" + str(j) + "x", "x" + str(j) + " * x"
            )
            temp_rounded_string = temp_rounded_string.replace(
                "x" + str(j) + "x", "x" + str(j) + " * x"
            )
            temp_rounded_string = temp_rounded_string.replace(
                "x" + str(j) + "x", "x" + str(j) + " * x"
            )
            temp_rounded_string = temp_rounded_string.replace(
                "x" + str(j) + "x", "x" + str(j) + " * x"
            )
        sym_equations_formatted.append(temp_string)
        sym_equations_rounded_formatted.append(temp_rounded_string)
    # Now that the equations are mathematically formatted, 
    # solve for x_dot0 in the algebraic equation.
    sym_equations_simplified = []
    sym_equations_rounded_simplified = []
    for i in range(nfeatures):
        print(i)
        sym_equations_simplified.append(
            sp.factor(sp.solve(
                sp.Add(
                    sp.sympify(sym_equations_formatted)[i][0],
                    -sp.sympify(features_formatted[i]),
                ),
                x_dot_sym[PC_index],
            ))
        )
        rounded = sp.factor(sp.solve(
            sp.Add(
                sp.sympify(sym_equations_rounded_formatted)[i][0],
                -sp.sympify(features_formatted[i]),
            ),
            x_dot_sym[PC_index],
        ))
        if len(rounded) != 0:
            rounded_temp = rounded[0]
            for a in sp.preorder_traversal(rounded):
                if isinstance(a, sp.Float):
                    rounded_temp = rounded_temp.subs(a, round(a, 2))
            sym_equations_rounded_simplified.append(rounded_temp)
        else:
            sym_equations_rounded_simplified.append([])
    return [sym_equations_simplified, sym_equations_rounded_simplified]

#####################################################################################
# preserved model trajectory grapher code
t_sym = sp.symbols("t_sym")
x_sym = sp.symbols("x:%d" % nPCs)
x_dot_sym = sp.symbols("x:%d_dot" % nPCs)
n_of_model = 0
# Plot the results for each of the models
plt.figure(figsize=(16, 10))
#x0_test = x_test[n_of_model][0, :]
for i in range(nPCs):
    plt.scatter(tvals, x_test[0][:,i], sizes=[20], label="True PC" + str(i))
for i in range(len(Sym_eqns[0][0])): #need to generalize for all Sym_eqns[all indexes]
    ax = plt.gca()
    #if i != nfeatures - 1:
        #ax.set_xticklabels([])
    if (len(Sym_eqns[0][0][i]) != 0 
        and len(Sym_eqns[1][0][i]) != 0 
        and len(Sym_eqns[2][0][i]) != 0):
        ODE_Func = lambda t, x: np.array([sp.lambdify(x_sym, Sym_eqns[0][0][i][0])(x[0], x[1], x[2]),
                                           sp.lambdify(x_sym, Sym_eqns[1][0][i][0])(x[0], x[1], x[2]),
                                           sp.lambdify(x_sym, Sym_eqns[2][0][i][0])(x[0], x[1], x[2])
                                         ])
        # Now simulate the system we identified
        print(f'solving model {i}')
        x_test_sim = solve_ivp(ODE_Func, (tvals[0], tvals[-1]), x0_test, t_eval=tvals).y.T
        if (
            np.linalg.norm(x_test_sim) < 1e3
            and Sym_eqns[0][1][i] != 0
            and Sym_eqns[1][1][i] != 0
            and Sym_eqns[2][1][i] != 0 #need to do for all Sym_eqns[all indexes]
        ):
            plt.plot(
                tvals,
                x_test_sim, # sim_data[n_of_model][i]
                linestyle="dashed"#,
                #label=str(sp.sympify(u_features_formatted[i]))
                #+ " = "
                #+ str(u_sym_equations_rounded_simplified[i]),
            )
# at the end so markers go over the lines

plt.grid(True)
#ax.set_ylim([0, 2])
plt.legend(fontsize=8)
plt.xlabel('Time', fontsize=15)
plt.title('Analysis of models for 3 principal components', fontsize=15)

#####################################################################################
# preserved error grapher code
plt.figure(figsize=(16, 10))
t_sym = sp.symbols("t_sym")
x_sym = sp.symbols("x:%d" % nPCs)
x_dot_sym = sp.symbols("x:%d_dot" % nPCs)

# number of models. Bad practice to use first PC tho
n_eqn = len(Sym[0][0])

# array to hold errors. Certain shape so we can index errors appropriately
error_per_model = np.empty((n_eqn))

# go thru # of models
for i in range(n_eqn):
    # if models exist
    if (len(Sym_eqns[0][0][i]) != 0 
        and len(Sym_eqns[1][0][i]) != 0 
        and len(Sym_eqns[2][0][i]) != 0):
        ODE_Func = lambda t, x: np.array([sp.lambdify(x_sym, Sym_eqns[0][0][i][0])(x[0], x[1], x[2]),
                                           sp.lambdify(x_sym, Sym_eqns[1][0][i][0])(x[0], x[1], x[2]),
                                           sp.lambdify(x_sym, Sym_eqns[2][0][i][0])(x[0], x[1], x[2])
                                         ])
        if (
            np.linalg.norm(x_test_sim) < 1e3 # if models are nontrivial
            and Sym_eqns[0][1][i] != 0
            and Sym_eqns[1][1][i] != 0
            and Sym_eqns[2][1][i] != 0
        ):
            print(f'solving model {i}')
            # error for some sample
            error_per_test_sample = []
            for j in range(len(x_test)):
                real = x_test[j]
                # take initial state of sample
                IC = x_test[j][0, :]
                x_test_sim = solve_ivp(ODE_Func, (tvals[0], tvals[-1]), IC, t_eval=tvals, **integrator_keywords).y.T
                MSE = (sum(np.square(np.subtract(real, x_test_sim)))) / (n_of_t*nPCs) # divide by dimension and timepoints
                error_per_test_sample.append(MSE)
            averaged_MSE = np.mean(error_per_test_sample)
            error_per_model[i] = averaged_MSE
plt.scatter(n_eqn,
            error_per_model,
            sizes=[20]
           )
plt.xlabel('Model #', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.title('Error per pySINDy-generated Model', fontsize=20)
