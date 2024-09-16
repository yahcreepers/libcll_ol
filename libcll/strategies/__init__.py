from .Strategy import Strategy
from .SCL import SCL
from .URE import URE
from .MCL import MCL
from .FWD import FWD
from .DM import DM
from .CPE import CPE
from .FreeMatch import FreeMatch
from .FixMatch import FixMatch
from .CL_FreeMatch import CL_FreeMatch
from .CL_FixMatch import CL_FixMatch

STRATEGY_LIST = {
    "SCL": SCL,
    "URE": URE,
    "MCL": MCL,
    "FWD": FWD,
    "DM": DM,
    "CPE": CPE,
    "FreeMatch": FreeMatch, 
    "FixMatch": FixMatch, 
    "CL_FreeMatch": CL_FreeMatch, 
    "CL_FixMatch": CL_FixMatch, 
}


def build_strategy(strategy, **args):
    if strategy not in STRATEGY_LIST:
        raise ValueError(f"Strategy must be chosen from {list(STRATEGY_LIST.keys())}.")
    return STRATEGY_LIST[strategy](**args)
