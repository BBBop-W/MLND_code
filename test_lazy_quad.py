import gurobipy as gp
from gurobipy import GRB

m = gp.Model()
x = m.addVar(vtype=GRB.BINARY, name="x")
m.setParam("LazyConstraints", 1)

def mycb(model, where):
    if where == GRB.Callback.MIPSOL:
        v = model.cbGetSolution(x)
        if v < 0.5:
            model.cbLazy(x * x >= 1)

m.optimize(mycb)
