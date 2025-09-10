def knapsack_pyomo_no_pruned_blocks(weight_dict, value_dict, channel_block_list, channel_group_name, channel_layer_name, layer2group, layer2block, channel_group_count, groups, capacity, extra_space, layer_index_split, results_dir, init_values=None):
    print("Running Pyomo")
    print('++++++')

    ori_capacity = capacity
    capacity += extra_space

    # Check if the value (importance) of neuron groups from the same layer have been
    # sorted ascendingly.
    model = ConcreteModel()
    # Define Decision Variables
    group_var_slices = {}
    counter = 0
    for group_name, value in channel_group_count.items():
        # print(group_name)
        group_var_slices[group_name] = (counter, counter+value)
        counter += value
    
    all_items = list(range(counter))
    # model.decision_vars = Var(all_items, domain=Binary)
    if init_values is not None:    
        model.decision_vars = Var(all_items, domain=Binary, initialize=init_values)
    else:
        model.decision_vars = Var(all_items, domain=Binary)
    # Add constraints
    # 1. Latency constraint: total latency need to be under the given budget;
    # 2. No layer prune constraint: Keep the most important neuron of a layer to avoid pruning
    # the entire layer.
    # 3. Only one unique configuration for one group
    latency_expr = 0
    importance = 0
    model.no_layer_prune_constraint = ConstraintList()
    model.group_unique_constraint = ConstraintList()
    for group_name in channel_group_count.keys():
        # Get the decision variables
        cur_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[group_name][0], group_var_slices[group_name][1])]
        # Unique Constraint. We can only select one configuration.
        model.group_unique_constraint.add(sum(cur_decision_vars[i] for i in list(range(len(cur_decision_vars)))) == 1)
        model.no_layer_prune_constraint.add(cur_decision_vars[0] == 0)
        for layer_name in group_name:
            # Latency Expression
            if layer_name == "module.conv1":
                pre_group_name, lat_vec = weight_dict[layer_name]
                latency_expr += sum(lat_vec[i] * cur_decision_vars[i] for i in range(len(cur_decision_vars)))
                model.first_layer_no_prune = Constraint(expr=cur_decision_vars[-1] == 1)
                continue
            pre_group_name, lat_matrix = weight_dict[layer_name]
            # pre_decision_vars = model.decision_vars[group_var_slices[pre_group_name][0]:group_var_slices[pre_group_name][1]]
            pre_decision_vars = [model.decision_vars[k] for k in range(group_var_slices[pre_group_name][0], group_var_slices[pre_group_name][1])]
            # lat_matrix: 1st dimension current group number, 2nd dimension previous group
            # channel_group_count x pre_channel_group_count
            # cur_decision_vars x (lat_matrix x pre_decision_vars.T)
            cur_expr = 0
            assert lat_matrix.shape[0] == len(cur_decision_vars)
            for i in range(len(cur_decision_vars)):
                cur_expr += cur_decision_vars[i] * sum(lat_matrix[i, j] * pre_decision_vars[j] for j in range(len(pre_decision_vars)))
            latency_expr += cur_expr
            
            # Importance Expression
            importance += sum(value_dict[layer_name][i]*cur_decision_vars[i] for i in list(range(len(cur_decision_vars))))

    # Latency Constraint
    model.latency_constraint = Constraint(expr=latency_expr <= capacity)
    
    # Set objective
    model.obj = Objective(expr=importance, sense=maximize)

    # Solve!
    # Mixed-Integer Nonlinear
    # model.obj.display()
    # model.display()
    # model.pprint()
    start = time.time()
    solver = SolverFactory('mindtpy')
    # solver = SolverFactory('glpk')
    # Without FP, problem is not solvable
    # if tries % 2 == 0:
    #     results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt')
    # else:
    #     results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt')

    results = solver.solve(model, strategy='OA', init_strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    # Initializing variable manually and search with FP is actually better for our usecase
    # results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    
    end = time.time()
    # results = solver.solve(model, strategy='FP', mip_solver='glpk', nlp_solver='ipopt') 
    # results = solver.solve(model) 
    print("Objective value:", model.obj())
    # print("Status = %s" % results.termination_condition)
    num_channel_to_keep = {}
    items_in_bag = []
    # Post-process the results
    # Check how many number of channels to keep for each layer 
    for group_name in channel_group_count.keys():
        indices = list(range(group_var_slices[group_name][0], group_var_slices[group_name][1]))
        cur_decision_vars = [model.decision_vars[k] for k in indices]
        for i in range(len(cur_decision_vars)):
            # print(cur_decision_vars[i].value)
            # How many groups to keep?
            if cur_decision_vars[i].value == 1:
                if i == 0:
                    print(f"No channels for group {group_name}")
                    break
                num_channel_to_keep[group_name] = i
    print("num_channel_to_keep")
    # print(num_channel_to_keep)
    # Get the indices for those kept channel groups
    cur_layer_name = None
    cur_start = None
    for i, layer_name in enumerate(channel_layer_name):
        if i == 0:
            cur_layer_name = layer_name
            cur_start = 0
        if layer_name != cur_layer_name:
            cur_end = i
            # They should come from the same group
            cur_groups = channel_group_name[cur_start:cur_end]
            assert len(set(cur_groups)) == 1
            cur_group = cur_groups[0]
            num_to_keep = num_channel_to_keep[cur_group]
            # Override by pruned block
            keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
            items_in_bag += keep_idxs
            # Update layer in consideration
            cur_layer_name = layer_name
            cur_start = i 
    cur_end = len(channel_layer_name)
    cur_groups = channel_group_name[cur_start:cur_end]
#         print(cur_groups)
#             print(cur_groups)
    assert len(set(cur_groups)) == 1
    cur_group = cur_groups[0]
    num_to_keep = num_channel_to_keep[cur_group]
    # Override by pruned block
    keep_idxs = list(range(cur_start, cur_end))[-num_to_keep:]
    items_in_bag += keep_idxs
        
    print("Solve Time", end-start)
    print("Finished")
    # exit()
    # for group_name, value in value_dict.items():
    #     print(group_name)
    #     print(model.decision_vars[group_var_slices[group_name][0]:group_var_slices[group_name][1]])
    # print(items_in_bag)
    return items_in_bag, None, None