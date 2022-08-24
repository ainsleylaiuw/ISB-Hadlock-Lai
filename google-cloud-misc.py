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