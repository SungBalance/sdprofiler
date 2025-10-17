

class A:
    def __init__(self, a: int):
        self.a = a
        self.setup()

    def setup(self):
        self.b = 2

class B(A):
    def __init__(self, a: int):
        super().__init__(a)

    def setup(self):
        self.b = 5

a = A(1)
print(a.a)

b = B(1)
print(b.a)
print(b.b)