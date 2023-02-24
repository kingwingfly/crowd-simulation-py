from collections import namedtuple
from torch import tensor
from torch.cuda import is_available as is_cuda_available
import json
import logging


device = "cuda" if is_cuda_available() else "cpu"
lr = 0.01  # Learning rate
Point = namedtuple('Point', ['row', 'col'])
Position = namedtuple('Position', ['floor_num', 'row', 'col'])


class Person:
    # todo 结伴
    def __init__(self, room: "Room") -> None:
        self.route = []
        self.cost = 0
        self._room = room  # The room contains position and exits info

    @property
    def room(self):
        """The room object where the person in"""
        return self._room

    @room.setter
    def room(self, target: "Room"):
        self._room = target

    @property
    def destination(self):
        """The destination Room the person wanna go

        Returns:
            Room object
        """
        return self.room.floor.building.get_room(Position(0, 0, 0))

    @property
    def exits(self):
        """The exits the room person in has

        Returns:
            [Room]
        """
        return self.room.exits

    @property
    def p(self):
        return tensor([1 / len(self.exits) for _ in self.exits], requires_grad=True)

    @property
    def position(self) -> Position:
        """The coordinate of the room the person in

        Returns:
            Position(floor_num: int, row: int, col: int)
        """
        return self.room.position

    def move(self, target: "Room") -> None:
        """Move the person to a certain room

        Args:
            target (Room): The target room
        """
        self.route.append(target.position)
        self.room = target
        self.cost += 1


class Exit:
    def __init__(
        self, outset: "Room", target: "Room", pass_factor: int | float
    ) -> None:
        """init an exit

        Args:
            outset (Room): The room where the exit belongs to
            target (Room): The room where the exit goes to
            pass_factor (int | float): A factor for the pass rate of the exit
        """
        self._pass_factor = pass_factor
        self._outset = outset
        self._target = target
        self.cross = False  # personnel cross-flow influence the pass_factor

    @property
    def pass_factor(self):
        """The factor that reflect the export capacity, if people cross, the bigger factor the better export capacity, the pass_factor will be smaller

        Returns:
            int | float: The factor that reflect the export capacity
        """
        return self._pass_factor * 0.7 if self.cross else self._pass_factor


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
        self._exits = None

    @property
    def exits(self):
        """The room's Exit objects

        Returns:
            [Exit]: The room's Exit objects
        """
        if not self._exits:
            self._exits = [
                Exit(
                    outset=self,
                    target=self.floor.get_room(
                        (self.position.row + row, self.position.col + col)
                    ),  # Room
                    pass_factor=1,
                )
                for row, col in self._exits
            ]
        return self._exits

    @property
    def reduce_rate(self) -> int | float:
        """The reduce rate of the room

        Returns:
            int | float: The reduce rate of the room
        """
        return sum(
            [
                1 / (self.population / self.area) * exit.pass_factor
                if self.population
                else float("inf")
                for exit in self.exits
            ]
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
    def position(self) -> Position:
        """The position of the room: (floor_num, row, col)

        Returns:
            Psition(floor_num: int, row: int, col: int)
        """
        return Position(
            floor_num=self.floor.floor_num, row=self._position[0], col=self._position[1]
        )


class Floor:
    def __init__(
        self,
        building: "Building",
        floor_num: int,
        floor_layout: dict[str, dict[str, str | int | float]],
    ) -> None:
        self._floor_num = floor_num
        self._map: dict[int, dict[int, Room]] = {}
        logging.info(f"The floor layout of F{self.floor_num} is\n{floor_layout}\n")
        for room_properties in floor_layout.values():
            room = Room(self, **room_properties)
            self._map[room.position.row] = self._map.get(room.position.row, {}) | {
                room.position.col: room
            }
        logging.info(f"The population distribution array is\n{self._map}\n")
        self._building = building
        self._population_distribution_tensor = None
        self._reduce_rates_tensor = None

    def __str__(self) -> str:
        return str(self.population_distribution)

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
        if not self._population_distribution_tensor:
            self._population_distribution_tensor = tensor(
                [
                    [
                        room.population
                        for _, room in sorted(row.items(), key=lambda x: x[0])
                    ]
                    for _, row in sorted(self._map.items(), key=lambda x: x[0])
                ],
                device=device,
            )
        logging.info(
            f"The population distribution is\n{self._population_distribution_tensor}\n"
        )
        return self._population_distribution_tensor

    @property
    def reduce_rates(self):
        """The tensor of the reduce rate distribution

        Returns:
            Tensor
        """
        if not self.reduce_rates:
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
        logging.info(f"The reduce rates are\n{self._reduce_rates_tensor}\n")
        return self._reduce_rates_tensor

    def get_room(self, position: Point) -> Room:
        """Get the Room object by position

        Args:
            position (Point[x: int, y: int]): (row, col)

        Returns:
            Room
        """
        if len(position) != 2:
            logging.error(f'The position need 2 figures but got {len(position)}')
        return self._map[position.row][position.col]


class Building:
    def __init__(self, floor_layouts) -> None:
        self._floors = self._floors_gen(floor_layouts=floor_layouts)

    def __str__(self) -> str:
        return '\n'.join([f"{floor}" for floor in self.floors])

    @property
    def floors(self):
        """Get Floor objects of the building

        Returns:
            [Floor]
        """
        return self._floors

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
            position (tuple[int, int, int]): (floor_num, row, col)

        Returns:
            Room
        """
        if len(position) != 3:
            logging.error(f'The position need 3 figures but got {len(position)}')
        return self.floors[position.floor_num].get_room((position.row, position.col))


class Frame:
    def __init__(self, building: Building) -> None:
        self.building = building

    def next_frame(self) -> "Frame":
        """Predict the next Frame for the later animtion and iteration"""
        ...

    def _forward(self):
        ...


if __name__ == "__main__":
    logging.basicConfig(
        filename='utils.log', filemode='w', encoding='utf-8', level=logging.DEBUG
    )
    with open("./floor_layouts.json", "r", encoding="utf-8") as f:
        floor_layouts = json.load(f)
    logging.debug(f"{floor_layouts}\n")
    building = Building(floor_layouts=floor_layouts)
    print(building)
