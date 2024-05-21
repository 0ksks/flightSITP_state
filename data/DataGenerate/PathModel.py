import numpy as np

np.set_printoptions(precision=3)


def get_plane_cost(totalRouteNum: int, availableRouteNum: int, costRange: tuple):
    cost = np.random.randint(*costRange, availableRouteNum)
    cost = np.append(cost, np.zeros(totalRouteNum - availableRouteNum))
    np.random.shuffle(cost)
    return cost


def get_route_cost(planeNum: int, routeNum: int, costRange=(10, 50)):
    mat = []
    for _ in range(planeNum):
        mat.append(
            get_plane_cost(
                routeNum, np.random.randint(int(0.3 * routeNum), routeNum), costRange
            )
        )
    return np.array(mat)


def get_cancel_cost(routeCost: np.ndarray, costMatter=True, flightCode=None):
    if not flightCode:
        routeNum = routeCost.shape[1]
        flightNum = np.random.randint(int(0.2 * routeNum), routeNum)
        flightCode = np.random.choice(range(flightNum), size=routeNum, replace=True)
    else:
        flightNum = max(flightCode) + 1
    costMax = routeCost.max(axis=0)
    flightCost = map(
        lambda code: costMax[np.where(flightCode == code)[0]].max(),
        np.unique(flightCode),
    )
    costMax = {k: v for k, v in zip(np.unique(flightCode), flightCost)}
    scale = 1.5 + np.random.rand() * 0.5 if costMatter else 1
    cancelCost = np.array(list(map(lambda k: scale * costMax[k], flightCode)))
    return flightCode, cancelCost


def get_data(
    planeNum: int,
    routeNum: int,
    costRange=(10, 50),
    costMatter=True,
    flightCode=None,
    dtype=np.float16,
) -> np.ndarray:
    routeCost = get_route_cost(planeNum, routeNum, costRange)
    flightCode, cancelCost = get_cancel_cost(routeCost, costMatter, flightCode)
    return np.append(routeCost, [flightCode, cancelCost], axis=0).astype(dtype)


if __name__ == "__main__":
    data = get_data(10, 5)
    print(data)
