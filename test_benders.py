"""Quick Benders test on scale_20."""
import sys
sys.path.insert(0, '.')
from src.data_processor.loader import DataLoader
from src.model.benders_model import BendersDecomposition
from src.model.gurobi_model import MLNDModel

inst = DataLoader().load_instance(
    'results/experiment_meituan/scale_20/locations.csv',
    'results/experiment_meituan/scale_20/orders.csv'
)

bd = BendersDecomposition(inst, max_iter=30, verbose=True,
                          time_limit=1200, master_time_limit=300)
bd.solve()

sol = bd.get_solution()
if sol:
    print(f"\ngamma = {sol['gamma']:.2f}")
    print("Open nodes:", [n for n, v in sol['x'].items() if v > 0.5])

grb = MLNDModel(inst)
grb.build()
grb.model.setParam('TimeLimit', 1200)
grb.model.setParam('MIPGap', 0.05)
grb.solve()
