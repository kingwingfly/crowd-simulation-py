from collections import namedtuple
from itertools import chain
from torch import tensor, Tensor
from torch.nn.functional import normalize
from torch.cuda import is_available as is_cuda_available
import json
import logging
from queue import Queue  # for storaging the move action of persons
from queue import PriorityQueue
from random import random
import matplotlib.pyplot as plt
import math

device = "cuda" if is_cuda_available() else "cpu"
lr = 1  # Learning rate
habit_factor = 0.25  # between [0, 1], indicates the degree to which people's choices are influenced by habits
destance_factor = 0.45  # between [0, 1], indicates the degree to which people's choices are influenced by destance to destinations
immdediate_factor = 0.3  # between [0, 1], indicates the degree to which people's choices are influenced by immediate situation
epoch_num = 100  # epoch number
Point = namedtuple('Point', ['row', 'col'])
Position = namedtuple('Position', ['floor_num', 'row', 'col'])


class Person:
    # todo 结伴
    def __init__(self, room: "Room") -> None:
        self._room = room  # The room contains position and exits info
        self._original_room = room
        self._route: list['Exit'] = []
        self._p: dict[int, dict[int, dict[int, Tensor]]] = {}

    def __str__(self) -> str:
        # todo
        return f"Position: {self.position} P: {self.p}"

    @property
    def route(self):
        return self._route

    @property
    def cost(self):
        return len(self.route)

    @property
    def room(self):
        """The room object where the person in"""
        return self._room

    @room.setter
    def room(self, target: "Room"):
        self._room._persons.remove(self)
        self._room = target
        target._persons.append(self)

    @property
    def floor(self):
        return self.room.floor

    @property
    def building(self):
        return self.room.building

    @property
    def exits(self):
        """The exits the room containing the person has

        Returns:
            list[Room]
        """
        return self.room.exits

    def get_p(self, position: Position):
        return (
            self._p.get(position.floor_num, {})
            .get(position.row, {})
            .get(position.col, None)
        )

    def set_p(self, position: Position, target: Tensor):
        """Normalize the target and set it as the p at the position given

        Args:
            position (Position): The position of p need change
            target (Tensor): p_tensor
        """
        self._p[position.floor_num][position.row][position.col] = normalize(
            target, p=1, dim=0
        )

    def modify_p(self, exit: 'Exit', exit_index, loss):
        """modify the p of the room where the exit is

        Args:
            exit (Exit): exit
            exit_index (_type_): the index of the exit's p
            loss (_type_): the loss * lr value
        """
        position = exit.outset.position
        old_value = self._p[position.floor_num][position.row][position.col][exit_index]
        # print(f'Old: {self._p[position.floor_num][position.row][position.col]}')
        new_value = max(old_value - loss, 0.001)
        self._p[position.floor_num][position.row][position.col][exit_index] = new_value
        self.set_p(position, self._p[position.floor_num][position.row][position.col])
        # print(f'New: {self._p[position.floor_num][position.row][position.col]}')

    @property
    def p(self):
        """The result is determined by habit, immediate decision and destance to the destinations.
        The habit could be trained, and is determined by the last iteration.
        The immediate decision is determined by the reduce rate of the exits which are at the same room as this exit.
        The destance is calculated by Dijkstra

        Returns:
            Tensor: sum() = 1
        """
        p = self.get_p(self.position)
        p_immediate = normalize(
            tensor(
                [exit.reduce_rate for exit in self.exits],
                device=device,
            ),
            p=1,
            dim=0,
        )  # todo consider stairs
        p_destance = normalize(
            tensor(
                [
                    1 / exit.target.destance if exit.target.destance else 1000.0
                    for exit in self.exits
                ],
                device=device,
            ),
            p=1,
            dim=0,
        )
        # todo This can be trained
        if p is None:
            self._p[self.position.floor_num] = self._p.get(self.position.floor_num, {})
            self._p[self.position.floor_num][self.position.row] = self._p[
                self.position.floor_num
            ].get(self.position.row, {})
            self.set_p(
                position=self.position,
                target=tensor([1 / len(self.exits) for _ in self.exits], device=device),
            )
        p_result = normalize(
            self.get_p(position=self.position) * habit_factor
            + p_destance * destance_factor
            + p_immediate * immdediate_factor,
            p=1,
            dim=0,
        )
        # todo annotate this when publish
        # if self.position == Position(1, 1, 1):
        #     print(p)
        # print([exit.target.position for exit in self.exits])
        return p_result

    @property
    def position(self) -> Position:
        """The coordinate of the room containing the person

        Returns:
            Position(floor_num: int, row: int, col: int)
        """
        return self.room.position

    @property
    def route(self):
        return self._route

    def whether_move(self) -> bool:
        """To judge if the person could move to next room

        Returns:
            bool: True for move to next room, False for keep the old room
        """
        return (
            True
            if self.room not in self.building.destinations
            and random() < self.room.reduce_rate / self.room.population
            else False
        )

    def where_move(self) -> 'Exit':
        """To determine which room the person will go to the next frame

        Returns:
            Room: The room the person will go to the next frame
        """
        r = random()
        ps = self.p
        for i in range(len(self.exits)):
            p = ps[i]
            if r - p <= 0 and p != 0:
                target = self.exits[i]
                logging.debug(
                    f'Person move from {self.position} to {target.target.position}'
                )
                return target
            r -= p

    def move(self, exit: "Exit") -> None:
        """Move the person to a certain room

        Args:
            target (Room): The target room
        """
        self._route.append(exit)
        self.room = exit.target

    def show_route(self):
        t = 'A lost person:\n'
        for exit in self.route:
            t += f'{exit.outset.position} =>\n'
        t += f'{self.route[-1].target.position}\n'
        logging.debug(t)

    def reset(self):
        """Reset this person to the original room and clean the cost and route but keep p"""
        self.room = self._original_room
        self._route = []


class Exit:
    def __init__(
        self,
        building: 'Building',
        outset: "Room",
        target: "Room",
        pass_factor: int | float,
    ) -> None:
        """init an exit

        Args:
            outset (Room): The room which the exit belongs to
            target (Room): The room which the exit goes to
            pass_factor (int | float): A factor for the pass rate of the exit
        """
        self._building = building
        self._pass_factor = pass_factor
        self._outset = outset
        self._target = target
        self.cross = False  # personnel cross-flow influence the pass_factor

    def __str__(self):
        return f"Exit: From {self.outset.position} to {self.target.position}"

    @property
    def building(self):
        return self._building

    @property
    def pass_factor(self):
        """The factor that reflect the export capacity, if people cross, the bigger factor the better export capacity, the pass_factor will be smaller

        Returns:
            int | float: The factor that reflect the export capacity
        """
        return self._pass_factor * 0.7 if self.cross else self._pass_factor

    @property
    def reduce_rate(self):
        """The passability, nothing to do with the people's choice,
        but the density of the people in both outset and target

        Returns:
            float: the number of person reduce through this exit per frame
        """
        population = (
            self.outset.population + self.target.population
            if self.target._exits_positions
            else 0.001
        )  # 如果通向建筑出口或者楼梯，人数视作0.001
        area = self.outset.area + self.target.area
        return 1 / (population / area) * self.pass_factor if population else 1000

    @property
    def target(self):
        return self._target

    @property
    def outset(self):
        return self._outset


class Room:
    def __init__(
        self,
        floor: "Floor",
        position: tuple[int, int],
        population: int,
        area: int | float,
        exits: list[tuple[int, int]],
    ) -> None:
        self._floor = floor
        self._position = Point(row=position[0], col=position[1])
        self._persons = [Person(room=self) for _ in range(population)]
        self._area = area
        self._exits_positions = exits  # The exits position
        self._exits = []

    def __str__(self):
        return f"Room: Position: {self.position}\nPopulation: {self.population}\nExits number: {len(self.exits)}\nReduce rate: {self.reduce_rate}"

    def __iter__(self):
        for person in self._persons:
            yield person

    @property
    def exits(self):
        """The room's Exit objects

        Returns:
            [Exit]: The room's Exit objects
        """
        if not self._exits:
            self._exits = (
                [
                    Exit(
                        building=self.building,
                        outset=self,
                        target=self.floor.get_room(
                            Point(
                                self.position.row + delta_row,
                                self.position.col + delta_col,
                            )
                        ),  # Room
                        pass_factor=1,
                    )
                    for delta_row, delta_col in self._exits_positions
                ]
                if self._exits_positions
                else [
                    Exit(
                        building=self.building,
                        outset=self,
                        target=self.building.get_room(
                            Position(
                                floor_num=max(self.position.floor_num - 1, 0),
                                row=self.position.row,
                                col=self.position.col,
                            )
                        ),
                        pass_factor=1,
                    )
                ]
            )
        return self._exits

    @property
    def reduce_rate(self) -> int | float:
        """The reduce rate of the room, the sum of the exits reduce rate of the room

        Returns:
            int | float: The reduce rate of the room
        """
        return sum(
            [exit.reduce_rate for exit in self.exits]
        )  # The population reduce rate is related to the density of people, the pass_factor and number of exits, the unit is people per frame

    @property
    def population(self):
        """The number of the person in the room

        Returns:
            int
        """
        return len(self._persons)

    @property
    def area(self):
        return self._area

    @property
    def floor(self):
        """The Floor object the room on

        Returns:
            Floor
        """
        return self._floor

    @property
    def building(self):
        """The Building object the room in

        Returns:
            Building
        """
        return self.floor.building

    @property
    def position(self) -> Position:
        """The position of the room: (floor_num, row, col)

        Returns:
            Psition(floor_num: int, row: int, col: int)
        """
        return Position(
            floor_num=self.floor.floor_num,
            row=self._position.row,
            col=self._position.col,
        )

    @property
    def persons(self) -> list[Person]:
        return self._persons

    @property
    def destance(self) -> int:
        myqueue = PriorityQueue()
        myqueue.put((0, self.position))
        cost_so_far = {self.position: 0}
        while not myqueue.empty():
            current: Room = self.building.get_room(myqueue.get()[1])
            if current in self.building.destinations:
                break
            for next_room in [exit.target for exit in current.exits]:
                new_cost = cost_so_far[current.position] + 1
                if (
                    next_room.position not in cost_so_far
                    or new_cost < cost_so_far[next_room.position]
                ):
                    cost_so_far[next_room.position] = new_cost
                    myqueue.put((new_cost, next_room.position))
        return cost_so_far[current.position]


class Floor:
    def __init__(
        self,
        building: "Building",
        floor_num: int,
        floor_layout: dict[str, dict[str, str | int | float]],
    ) -> None:
        self._floor_num = floor_num
        self._map: dict[int, dict[int, Room]] = {}
        logging.debug(f"The floor layout of F{self.floor_num} is\n{floor_layout}\n")
        for room_properties in floor_layout.values():
            room = Room(floor=self, **room_properties)
            self._map[room.position.row] = self._map.get(room.position.row, {}) | {
                room.position.col: room
            }
        self._building = building
        self._population_distribution_tensor = None
        self._reduce_rates_tensor = None

    def __str__(self) -> str:
        return str(self.population_distribution)

    def __iter__(self):
        for row in self._map.values():
            for room in row.values():
                yield room

    @property
    def building(self):
        """The Building object the floor in

        Returns:
            Building
        """
        return self._building

    @property
    def floor_num(self):
        """The floor_num of the floor

        Returns:
            int
        """
        return self._floor_num

    @property
    def population_distribution(self):
        """The tensor of the population distribution

        Returns:
            Tensor
        """
        self._population_distribution_tensor = tensor(
            [
                [room.population for _, room in sorted(row.items(), key=lambda x: x[0])]
                for _, row in sorted(self._map.items(), key=lambda x: x[0])
            ],
            device=device,
        )
        logging.debug(
            f"The population distribution is\n{self._population_distribution_tensor}\n"
        )
        return self._population_distribution_tensor

    @property
    def reduce_rates(self):
        """The tensor of the reduce rate distribution

        Returns:
            Tensor
        """
        self._reduce_rates_tensor = tensor(
            [
                [
                    room.reduce_rate
                    for _, room in sorted(row.items(), key=lambda x: x[0])
                ]
                for _, row in sorted(self._map.items(), key=lambda x: x[0])
            ],
            device=device,
        )
        logging.debug(f"The reduce rates are\n{self._reduce_rates_tensor}\n")
        return self._reduce_rates_tensor

    @property
    def total_popularity(self) -> int:
        """The total people number of this floor

        Returns:
            int: Total people number of the floor
        """
        return sum([room.population for room in self])

    @property
    def persons(self):
        return chain(*[room.persons for room in self])

    def get_room(self, position: Point) -> Room:
        """Get the Room object by position

        Args:
            position (Point[row: int, col: int]): (row, col)

        Returns:
            Room
        """
        if len(position) != 2:
            logging.error(f'The position need 2 figures but got {len(position)}')
        return self._map[position.row][position.col]


class Building:
    def __init__(self, floor_layouts: dict) -> None:
        self._floors = self._floors_gen(floor_layouts=floor_layouts)

    def __str__(self) -> str:
        return '\n'.join([f"{floor}" for floor in self.floors]) + '\n'

    def __iter__(self):
        for floor in self.floors:
            yield floor

    @property
    def floors(self):
        """Get Floor objects of the building

        Returns:
            list[Floor]
        """
        return self._floors

    @property
    def total_popularity(self) -> int:
        """The total people number of this building

        Returns:
            int: Total people number of the building
        """
        return sum([floor.total_popularity for floor in self])

    @property
    def persons(self):
        return chain(*[floor.persons for floor in self])

    @property
    def destinations(self):
        """The destination Rooms of the building.

        Returns:
            Room object
        """
        # todo
        positions = [Position(0, 2, 1)]
        return [self.get_room(position) for position in positions]

    def _floors_gen(
        self, floor_layouts: dict[str, dict[str, dict[str, str | int | float]]]
    ) -> list[Floor]:
        """Generate a list of Floors according to the json dictionary

        Args:
            floor_layouts (dict[str, dict[str, dict[str, str  |  int  |  float]]]): A dict read from the json config: floor_layouts.json

        Returns:
            list[Floor]: A list of Floors, from 1st floor to last floor
        """
        return [
            Floor(building=self, floor_num=floor_num, floor_layout=floor_layout)
            for floor_num, floor_layout in enumerate(floor_layouts.values())
        ]

    def add_floor(self, floor: Floor | list[Floor]):
        """Add floor or floors to a Building Object

        Args:
            floor (Floor | list[Floor]): Both Floor and list of Floors is accessible.
        """
        if isinstance(floor, (list)):
            self._floors.extend(floor)
            return
        self._floors.append(floor)

    def get_room(self, position: Position) -> Room:
        """Get the Room object at the position passed in

        Args:
            position (Position[int, int, int]): (floor_num, row, col)

        Returns:
            Room
        """
        if len(position) != 3:
            logging.error(f'The position need 3 figures but got {len(position)}')
        return self.floors[position.floor_num].get_room(
            Point(position.row, position.col)
        )

    def reset(self):
        """Reset all people to their original room and clean their cost and route but keep p"""
        for person in list(self.persons):
            # For the self.persons would change, clone the person list first
            person.reset()


class Simulator:
    def __init__(self, building: Building) -> None:
        self._building = building
        self._queue = Queue()
        self._frame_num = 0

    @property
    def frame_num(self) -> int:
        return self._frame_num

    @property
    def building(self) -> Building:
        return self._building

    @property
    def persons(self) -> chain(Person):
        return self.building.persons

    def _reset(self):
        """Reset the simulator and the building but keep p"""
        self._frame_num = 0
        self.building.reset()

    def _person_move(self):
        """Move all persons in building according to person.p and reduce rate"""
        for person in self.persons:
            if not person.whether_move():
                continue
            self._queue.put((person, person.where_move()))

        while not self._queue.empty():
            person: Person
            target: Exit
            person, target = self._queue.get()
            person.move(target)

    def _next_frame(self):
        """Predict the next Frame for the later animtion and iteration"""
        self._person_move()
        self._frame_num += 1

    def forward(self):
        self._reset()
        while (
            self.building.get_room(Position(floor_num=0, row=2, col=1)).population
            != self.building.total_popularity
        ):
            self._next_frame()
        logging.info(f'Simulation ended within {self.frame_num} frames\n')
        return self.frame_num


class Optimizer:
    def __init__(self, building: Building, learning_rate: float) -> None:
        """optimizer"""
        self._building = building
        self._lr = learning_rate
        self.epoch = 0

    @property
    def lr(self):
        return self._lr - self._lr * (self.epoch - epoch_num / 2) / epoch_num

    def step(self, loss: float):
        loss *= self.lr if loss > 0 else self.lr * 10
        for person in self._building.persons:
            if person.cost > 5:
                person.show_route()
            for exit in person.route:
                i = exit.outset.exits.index(exit)
                person.modify_p(exit=exit, exit_index=i, loss=loss)
        self.epoch += 1

    def zero_grad(self):
        ...


class Criterion:
    def __init__(self) -> None:
        self.loss = 0

    def __call__(self, output, target):
        """Calculate loss

        Args:
            output (_type_): _description_
            target (_type_): _description_

        Returns:
            Criterion: self
        """
        if target == float("inf"):
            self.loss = 0
            return self.loss
        x = (output - target) / 10
        if x <= 0:
            self.loss = math.exp(x) - 1
        else:
            self.loss = 1 - 1.1 ** (-x)
        return self.loss


def train(epoch_num):
    building = Building(floor_layouts=floor_layouts)
    simulator = Simulator(building=building)
    criterion = Criterion()
    optim = Optimizer(building=building, learning_rate=lr)
    best = float("inf")
    train_info = []
    for epoch in range(epoch_num):
        output = simulator.forward()
        loss = criterion(output=output, target=best)
        best = min(best, output)
        optim.zero_grad()
        optim.step(loss)
        train_info.append((epoch, loss))
        logging.info(
            f"Finish epoch: {epoch}\twithin {output} frames; current best is {best}; loss is {loss}"
        )
        print(
            f"Finish epoch: {epoch}\twithin {output} frames; current best is {best}; loss is {loss}"
        )
    logging.info(f"The fastest reslut is {best} frames")
    print(f"The fastest reslut is {best} frames")
    return train_info


def draw_loss(train_info: list[(int, float)]):
    plt.xlabel = 'epoch_num'
    plt.ylabel = 'loss'
    epoch, loss = [i[0] for i in train_info[1::]], [i[1] for i in train_info[1::]]
    plt.scatter(epoch, loss)
    # plt.show()
    plt.savefig('./result.png')


if __name__ == "__main__":
    logging.basicConfig(
        filename='utils.log', filemode='w', encoding='utf-8', level=logging.INFO
    )
    with open("./floor_layouts.json", "r", encoding="utf-8") as f:
        floor_layouts = json.load(f)
    logging.debug(f"The floor layouts:\n{floor_layouts}\n")
    train_info = train(epoch_num)
    draw_loss(train_info=train_info)
    print("Finish!")
