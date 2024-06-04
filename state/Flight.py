from typing import NamedTuple
import networkx as nx  # 导入networkx库来操作图结构
from datetime import datetime, timedelta


class FlightInfo:
    def __init__(self, depPort, arrPort, depTime, arrTime, series, is_chosen=0):
        self.depPort = depPort
        self.arrPort = arrPort
        self.depTime = depTime
        self.arrTime = arrTime
        self.is_chosen = is_chosen  # 默认为0，表示未被选中
        self.series = series

    def __str__(self) -> str:
        depTime = str(self.depTime).replace("T", " ")
        depTime = depTime.replace(".000Z", "")
        arrTime = str(self.arrTime).replace("T", " ")
        arrTime = arrTime.replace(".000Z", "")
        fmtStr = "{} {}\n{} {}\n[{}]".format(
            self.depPort,
            depTime,
            self.arrPort,
            arrTime,
            self.series,
        )
        return fmtStr

    def flat_str(self) -> str:
        depTime = str(self.depTime).replace("T", " ")
        depTime = depTime.replace(".000Z", "")
        arrTime = str(self.arrTime).replace("T", " ")
        arrTime = arrTime.replace(".000Z", "")
        fmtStr = "({}, {}, {}, {})[{}]".format(
            self.depPort,
            depTime,
            arrTime,
            self.arrPort,
            self.series,
        )
        return fmtStr


class StateFS(NamedTuple):
    flights: list[FlightInfo]  # List of FlightInfo objects
    graph: nx.DiGraph  # Directed graph representing flight connections
    current_flight: FlightInfo = None  # Property to track the currently selected flight
    zero_in_degree_flights: list = []  # List to store flights with zero in-degree
    chains: list[list[FlightInfo]] = [[]]
    staytime: int = 0
    i: int = 0

    # 初始化航班状态和有向图
    @staticmethod
    def initialize(flights: list[FlightInfo]):
        graph = nx.DiGraph()
        for i in range(len(flights)):
            current_flight = flights[i]
            for j in range(i + 1, len(flights)):
                next_flight = flights[j]
                if (
                    current_flight.arrPort == next_flight.depPort
                    and current_flight.arrTime < next_flight.depTime
                ):
                    graph.add_edge(current_flight, next_flight)
        zero_in_degree_flights = [
            node for node, in_degree in list(graph.in_degree()) if in_degree == 0
        ]
        chains: list[list[FlightInfo]] = [[]]
        chains[0].append(zero_in_degree_flights[0])
        flights[0].is_chosen = 1
        return StateFS(
            flights=flights,
            graph=graph,
            zero_in_degree_flights=zero_in_degree_flights,
            current_flight=zero_in_degree_flights[0],
            chains=chains,
        )

    # 更新航班状态
    def update(self, selected_flight: FlightInfo, mask):
        # 从图中移除选中航班的出边，表示该航班已经被安排
        current_flight = self.current_flight  # 更新当前选中的航班
        out_edges = list(self.graph.out_edges(current_flight))
        graph = self.graph
        graph.remove_edges_from(out_edges)

        current_time = datetime.fromisoformat(current_flight.arrTime.replace("Z", ""))
        select_time = datetime.fromisoformat(selected_flight.depTime.replace("Z", ""))
        if mask == [0, 1]:
            staytime = self.staytime
        elif mask == [1, 0]:
            staytime = (select_time - current_time) // timedelta(
                minutes=1
            ) + self.staytime
        elif mask == [1, 1]:
            for i in range(0, len(self.chains)):
                last_flights = self.chains[i][-1]
                successors_list2 = list(self.graph.successors(last_flights))
                if selected_flight in successors_list2:
                    current_time = datetime.fromisoformat(
                        last_flights.arrTime.replace("Z", "")
                    )
                    break
            staytime = (select_time - current_time) // timedelta(
                minutes=1
            ) + self.staytime
        # 更新入度为零的航班列表
        zero_in_degree_flights = [
            node
            for node, in_degree in list(self.graph.in_degree())
            if in_degree == 0 and node.is_chosen == 0
        ]

        return self._replace(
            graph=graph,
            current_flight=selected_flight,
            zero_in_degree_flights=zero_in_degree_flights,
            staytime=staytime,
        )

    # 当航班被选中后，设置 is_chosen 为1，并更新航班状态
    def select_flight(self, flight):
        flight.is_chosen = 1
        mask = self.get_mask(flight)
        if mask == [0, 0]:
            flight.is_chosen = 0  # 将 is_chosen 设为 0，表示航班未被选中
            return False
        elif mask == [0, 1]:
            if flight in self.zero_in_degree_flights:
                flightschedule = self.update(flight, mask)
                # 创建新的链并添加选中的航班
                new_chain = [flight]  # 创建一个新的链列表，包含添加的新链
                i = self.i + 1
                new_chains = self.chains + [new_chain]
                flightschedule = flightschedule._replace(chains=new_chains, i=i)
                return flightschedule  # 使用 _replace 方法更新 chains 参数
            else:
                flight.is_chosen = 0  # 将 is_chosen 设为 0，表示航班未被选中
                return False
        elif mask == [1, 0]:
            flightschedule = self.update(flight, mask)  # 选择航班后更新航班状态
            # 创建新的链列表并复制 self.chains 中的内容
            new_chains = self.chains
            new_chains[self.i].append(flight)
            flightschedule = flightschedule._replace(chains=new_chains)
            return flightschedule  # 使用 _replace 方法更新 chains 参数
        elif mask == [1, 1]:
            for i in range(0, len(self.chains)):
                last_flights = self.chain[i][-1]
                successors_list2 = list(self.graph.successors(last_flights))
                if flight in successors_list2:
                    statefs = self.update(flight, mask)
                    new_chains = self.chains
                    new_chains[i].append(flight)
                    statefs = statefs._replace(chains=new_chains)

    def get_mask(self, selected_flight):
        """
        Determines whether the selection of the next flight is correct based on the following criteria:
        1. The next flight to be selected should be within the directed edge of the current flight.
        2. Checks if the current flight has a next flight; returns a flag if there is none.

        Args:
        current_flight (FlightInfo): The current flight that has been selected.

        """
        mask = [0, 0]
        successors_list = list(self.graph.successors(self.current_flight))
        if len(successors_list) > 0:
            if selected_flight in successors_list:
                mask = [1, 0]
        elif len(successors_list) == 0:
            for i in range(0, len(self.chains)):
                last_flights = self.chains[i][-1]
                successors_list2 = list(self.graph.successors(last_flights))
                if selected_flight in successors_list2:
                    mask = [1, 1]
                    break
                else:
                    mask = [0, 1]
        return mask
