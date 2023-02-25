class A:
    def __init__(self, b: 'B') -> None:
        self.b = b

    def foo(self):
        self.b._lst.remove(self)

class B:
    def __init__(self) -> None:
        self._lst = [A(self) for _ in range(6)]

    
b = B()
a = b._lst[0]
a.foo()