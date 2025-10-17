from dataclasses import dataclass

@dataclass
class A:
    a: int = 1

a = A()
b = a
b.a = 2

print(a)