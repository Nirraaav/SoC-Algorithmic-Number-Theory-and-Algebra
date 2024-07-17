from sage import *
from sage.all import *
import random
import soc24mathlib

a = random.randint(1, 100000000)
while a % 2 == 0:
    a = random.randint(1, 100000000)

p = soc24mathlib.gen_prime(1000000000000)
while p % 4 == 3:
    p = soc24mathlib.gen_prime(1000000000000)

try:
    sqrt_a = sqrt(Mod(a, p))
    assert sqrt_a**2 == Mod(a, p)
    int_sqrt_a = int(sqrt_a)
    print(int_sqrt_a)
    print(soc24mathlib.modular_sqrt_prime(a, p), int_sqrt_a)
    assert soc24mathlib.modular_sqrt_prime(a, p) == int_sqrt_a
except ValueError:
    print(f"No square root exists for a = {a} modulo p = {p}")