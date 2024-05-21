import numpy as np
from scipy.sparse import coo_matrix

def onehot2num(arr: np.ndarray) -> np.ndarray:
    return np.where(arr == 1)[1]


def solve(data: np.ndarray):
    routeCost = data[:-2, :]
    shape = routeCost.shape
    import pulp
    import re

    pathLP = pulp.LpProblem("PathModel", sense=pulp.LpMinimize)

    varDict = {}
    varList = []
    assignObj = 0
    for i in range(shape[0]):
        varRow = []
        for j in range(shape[1]):
            if routeCost[i][j] != 0:
                varDict[f"F{i}R{j}"] = pulp.LpVariable(f"F{i}R{j}", cat=pulp.LpBinary)
                varRow.append(varDict[f"F{i}R{j}"])
                assignObj += routeCost[i][j] * varDict[f"F{i}R{j}"]
        varList.append(varRow)
    # (13)define
    from collections import defaultdict

    cancel = defaultdict(list)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if routeCost[i][j] != 0:
                cancel[data[-2][j]].append(varDict[f"F{i}R{j}"])
    flightNum2CancelCost = {k: v for k, v in zip(data[-2], data[-1])}
    cancelZ = {}
    cancelObj = 0
    for k in flightNum2CancelCost.keys():
        k = int(k)
        cancelZ[f"Z{k}"] = pulp.LpVariable(f"Z{k}", cat=pulp.LpBinary)
        # cancel objective
        cancelObj += flightNum2CancelCost[k] * cancelZ[f"Z{k}"]

    # objective
    pathLP += (assignObj + cancelObj), "Objective"

    # (12) apply
    for idx, varRow in enumerate(varList):
        if len(varRow) == 1:
            pathLP += (varRow[0] <= 1), f"Constraint_12_{idx}"
        else:
            pathLP += (sum(varRow) <= 1), f"Constraint_12_{idx}"

    # (13) apply
    for idx, k in enumerate(flightNum2CancelCost.keys()):
        k = int(k)
        pathLP += (sum(cancel[k]) + cancelZ[f"Z{k}"] == 1), f"Constraint_13_{idx}"

    pathLP.solve()
    result = np.zeros(shape)
    for k, v in varDict.items():
        if v.varValue == 1:
            pos = [int(num) for num in re.findall(r"\d+", v.name)]
            result[pos[0]][pos[1]] = 1
    new_column = np.where(result.sum(axis=1) == 0, 1, 0)
    result = np.column_stack((result, new_column))
    result = onehot2num(result)
    data[[-1, -2]] = data[[-2, -1]]
    coo = coo_matrix(data[:-2])

    return {
        "input": (coo.data.tolist(), (coo.row.tolist(), coo.col.tolist())),
        "pos": data[-1].astype(int).tolist(),
        "shape": list(shape),
        "output": result.tolist(),
        "cost": pulp.value(pathLP.objective),
    }


if __name__ == "__main__":
    import sys

    sys.path.append("/Users/admin/Desktop/LEARN/大三下/FLIGHT")
    from data.DataGenerate.PathModel import get_data

    data = get_data(planeNum=6, routeNum=4)
    solution = solve(data)
    print(solution)
    print(coo_matrix(solution["input"], solution["shape"]).todense())
