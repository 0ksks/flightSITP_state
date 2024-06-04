import pandas as pd
from Flight import FlightInfo, StateFS

# 读取CSV文件并创建DataFrame
input_data = pd.read_csv("./schedule.csv")

# 将DataFrame转换为FlightInfo对象列表
flights = []
for _, row in input_data.iterrows():
    flight = FlightInfo(
        depPort=row["depPort"],
        arrPort=row["arrPort"],
        depTime=row["depTime"],
        arrTime=row["arrTime"],
        series=row["series"],
    )
    flights.append(flight)

# 初始化航班状态和有向图
flight_schedule = StateFS.initialize(flights)
# nx.nx_agraph.write_dot(
#     flight_schedule.graph, "./graph.dot"
# )  # comment this line if you don't want to draw a picture
# 选择航班并将更新后的航班状态赋值给新的变量
updated_flight_schedule = flight_schedule.select_flight(flights[1])
# 获取前驱航班列表
predecessors = list(updated_flight_schedule.graph.predecessors(flights[51]))
# 打印前驱航班信息
print(f"前驱航班 for {flights[51].series}:")
for predecessor in predecessors:
    print(
        f"Flight {predecessor.series} from {predecessor.depPort} to {predecessor.arrPort} departs at {predecessor.depTime} and arrives at {predecessor.arrTime}"
    )

failed_flight = []
success_flight = []
for i in range(2, 180):
    updated_flight_schedule = updated_flight_schedule.select_flight(flights[i])
    if updated_flight_schedule:
        success_flight.append(updated_flight_schedule)
    else:
        failed_flight.append(i)
        updated_flight_schedule = success_flight[-1]
predecessors1 = list(updated_flight_schedule.graph.predecessors(flights[51]))
# 打印前驱航班信息
print(f"前驱航班 for {flights[51].series}:")
for predecessor in predecessors:
    print(
        f"Flight {predecessor.series} from {predecessor.depPort} to {predecessor.arrPort} departs at {predecessor.depTime} and arrives at {predecessor.arrTime}"
    )

# 检查是否选择成功，并打印航班状态
if updated_flight_schedule:
    print("Flight selected successfully!")
    print("Current Flight:", updated_flight_schedule.current_flight.series)
    print("Chains:")
    for chain in updated_flight_schedule.chains:
        print("=" * 50)
        print("{}".format("\n".join(map(lambda x: x.flat_str(), chain))))
    print("=" * 50)
    print("staytime:", updated_flight_schedule.staytime)
else:
    print("Flight selection failed.")
