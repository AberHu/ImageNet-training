import os
import sys
from benchmark import ModelStat, stat
from benchmark import compute_speed

sys.path.append('..')
from networks import MobileNetV3_Small, MobileNetV3_Large


model = MobileNetV3_Small()

# query_granularity can be any int value, usually:
# query_granularity=1  reports every leaf node
# query_granularity=-1 only reports the root node
stat(model, (1, 3, 224, 224), query_granularity=1, brief_report=False)
stat(model, (1, 3, 224, 224), query_granularity=-1, brief_report=False)

# brief_report=True only reports the summation
stat(model, (1, 3, 224, 224), query_granularity=1, brief_report=True)


# can also initialize ModelStat, set the query_granularity and then show_report
ms = ModelStat(model, (1, 3, 224, 224), query_granularity=1, brief_report=False)

ms.query_granularity = -1
ms.show_report()
ms.query_granularity = 1
ms.show_report()

ms.unhook_model()

# measure latency
compute_speed(model, (32, 3, 224, 224), 'cuda:0', 1000)
compute_speed(model, (1, 3, 224, 224), 'cuda:0', 1000)
compute_speed(model, (1, 3, 224, 224), 'cpu', 1000)


#!!! there are 1 bug not fixed: MemWrite(B) has different values with different query_granularity
