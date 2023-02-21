from torch import tensor, zeros
from torch.cuda import is_available as is_cuda_available
import json

device = "cuda" if is_cuda_available() else "cpu"
lr = 0.01  # Learning rate


class Person:
    # todo 结伴
    def __init__(self, room: 'Room') -> None:
        self.route = []
        self.cost = 0
        self._room = room  # The room contains position and exits info

    @property
    def room(self):
        return self._room

    @room.setter
    def room(self, target: 'Room'):
        self._room = target

    @property
    def destination(self):
        ...
        return self.room.floor.building

    @property
    def exits(self):
        return self.room.exits

    @property
    def p(self):
        return [1 / len(self.exits) for _ in self.exits]

    def move(self, target: 'Room'):
        self.route.append(target.position)
        self.room = target
        self.cost += 1


class Exit:
    def __init__(
        self, outset: 'Room', target: 'Room', pass_factor: int | float
    ) -> None:
        """init an exit

        Args:
            outset (_type_): The room where the exit belongs to
            target (_type_): The room where the exit goes to
            pass_factor (_type_): A factor for the pass rate of the exit
        """
        self._pass_factor = pass_factor
        self._outset = outset
        self._target = target
        self.cross = False  # personnel cross-flow influence the pass_factor

    @property
    def pass_factor(self):
        return self._pass_factor * 0.7 if self.cross else self._pass_factor


class Room:
    def __init__(
        self,
        floor: 'Floor',
        position: tuple,
        population: int,
        area: int | float,
        exits: list[list[int]],
    ) -> None:
        self._floor = floor
        self._position = position
        self._persons = [Person(room=self) for _ in range(population)]
        self._area = area
        self._exits = exits

    @property
    def exits(self):
        return [
            Exit(
                outset=self,
                target=self._floor._map[self.position[0] + row][
                    self.position[0] + col
                ],  # Room
                pass_factor=1,
            )
            for row, col in self._exits
        ]

    @property
    def reduce_rate(self) -> int | float:
        return sum(
            [
                1 / (self.population / self.area) * exit.pass_factor
                if self.population
                else float('inf')
                for exit in self.exits
            ]
        )  # The population reduce rate is related to the density of people, the pass_factor and number of exits, the unit is people per frame

    @property
    def population(self):
        return len(self._persons)

    @property
    def area(self):
        return self._area

    @property
    def floor(self):
        return self._floor

    @property
    def position(self):
        return self._position


class Floor:
    def __init__(
        self,
        building: 'Building',
        floor_layout: dict[str, dict[str, str | int | float]],
    ) -> None:
        self._map: dict[int, dict[int, Room]] = {}
        for room_properties in floor_layout.values():
            room = Room(self, **room_properties)
            self._map[room.position[0]] = self._map.get(room.position[0], {}) | {
                room.position[1]: room
            }
        self._building = building

    @property
    def building(self):
        return self._building

    def population_distribution(self):
        self.population_distribution_tensor = tensor(
            [
                [room.population for _, room in sorted(row.items(), key=lambda x: x[0])]
                for _, row in sorted(self._map.items(), key=lambda x: x[0])
            ],
            device=device,
        )
        print(
            f"The population distribution is\n{self.population_distribution_tensor}\n"
        )
        return self.population_distribution_tensor

    def reduce_rates(self):
        self.reduce_rates_tensor = tensor(
            [
                [
                    room.reduce_rate
                    for _, room in sorted(row.items(), key=lambda x: x[0])
                ]
                for _, row in sorted(self._map.items(), key=lambda x: x[0])
            ],
            device=device,
        )
        print(f"The reduce rates are\n{self.reduce_rates_tensor}\n")
        return self.reduce_rates_tensor


class Building:
    def __init__(self, floor_layouts) -> None:
        self._floors = self.floors_gen(floor_layouts=floor_layouts)

    def floors_gen(
        self, floor_layouts: dict[str, dict[str, dict[str, str | int | float]]]
    ) -> list[Floor]:
        """Generate a list of Floors according to the json dictionary

        Args:
            floor_layouts (dict[str, dict[str, dict[str, str  |  int  |  float]]]): A dict read from the json config: floor_layouts.json

        Returns:
            list[Floor]: A list of Floors, from 1st floor to last floor
        """
        return [
            Floor(building=self, floor_layout=floor_layout)
            for floor_layout in floor_layouts.values()
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


class Frame:
    def __init__(self, building: Building) -> None:
        self.building = building

    def next_frame(self) -> 'Frame':
        """Predict the next Frame for the later animtion and iteration"""
        ...

    def _forward(self):
        ...


if __name__ == "__main__":
    with open('./floor_layouts.json', 'r', encoding='utf-8') as f:
        floor_layouts = json.load(f)
    building = Building(floor_layouts=floor_layouts)
