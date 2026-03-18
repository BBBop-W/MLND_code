"""Debug max_iter type issue."""
import sys
sys.path.insert(0, '.')

from src.data_processor.loader import DataLoader
from src.model.benders_primal import BendersPrimal

inst = DataLoader().load_instance(
    'data/instances/Meituan/locations.csv',
    'data/instances/Meituan/orders.csv',
)

primal = BendersPrimal(inst, max_iter=50, verbose=False)
print(f"max_iter={primal.max_iter}, type={type(primal.max_iter)}")
print(f"data={primal.data}, type={type(primal.data)}")
