from pyomo.environ import *

# Data
weights = [2, 3, 4, 5, 2, 1]  # Item weights
values = [3, 4, 5, 6, 3, 2]  # Item values
groups = [1, 2, 1, 3, 2, 3]  # Group numbers
capacity = 8  # Knapsack capacity

model = ConcreteModel()

# Decision variables
model.x = Var(range(len(weights)), domain=Binary)
model.y = Var(set(groups), domain=Binary)

# Objective function: Maximize value and minimize the number of selected groups
model.obj = Objective(expr=sum(values[i] * model.x[i] for i in range(len(weights))) - sum(model.y[g] for g in set(groups)), sense=maximize)

# Constraint: Knapsack capacity
model.capacity_constraint = Constraint(expr=sum(weights[i] * model.x[i] for i in range(len(weights))) <= capacity)

# Constraint: If a group is selected, at least one value from that group must be selected
model.group_constraint = ConstraintList()
for g in set(groups):
    items_in_group = [i for i in range(len(weights)) if groups[i] == g]
    model.group_constraint.add(sum(model.x[i] for i in items_in_group) >= model.y[g])

# Constraint: If a group is not selected, no value from that group must be selected
for g in set(groups):
    items_in_group = [i for i in range(len(weights)) if groups[i] == g]
    model.group_constraint.add(sum(model.x[i] for i in items_in_group) <= len(items_in_group) * model.y[g])

# Solve the problem
solver = SolverFactory('glpk')
results = solver.solve(model)

# Print the solution
print("Objective value:", model.obj())
print("Selected items:")
for i in range(len(weights)):
    if model.x[i].value == 1:
        print("Item", i, "(weight:", weights[i], ", value:", values[i], ", group:", groups[i], ")")
print("Selected groups:")
for g in set(groups):
    if model.y[g].value == 1:
        print("Group", g)