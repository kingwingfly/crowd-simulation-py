class Test:
    def __init__(self) -> None:
        self._x = 1

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, y):
        self._x = y


test = Test()
print(test.x)
test.x = 2
print(test.x)
