# # from sage import *
# # from sage.all import *
# import random
# import soc24mathlib

# while True:
#     a = random.randint(1, 1000000000000000)
#     while a % 2 == 0:
#         a = random.randint(1, 1000000000000000)

#     # p = soc24mathlib.gen_prime(1000000000000)
#     # while p % 4 == 3:
#     #     p = soc24mathlib.gen_prime(1000000000000)

#     b = random.randint(1, 100000000)

#     # try:
#     #     sqrt_a = sqrt(Mod(a, p))
#     #     assert sqrt_a**2 == Mod(a, p)
#     #     int_sqrt_a = int(sqrt_a)
#     #     print(int_sqrt_a)
#     #     print(a, p, soc24mathlib.modular_sqrt_prime(a, p), int_sqrt_a)
#     #     assert soc24mathlib.modular_sqrt_prime(a, p) == int_sqrt_a
#     # except ValueError:
#     #     print(f"No square root exists for a = {a} modulo p = {p}")

#     # try:
#     #     moda = soc24mathlib.mod_inv(a, b)
#     #     moda1 = soc24mathlib.mod_inverse(a, b)
#     #     print(moda, moda1, a, b)
#     #     assert moda == moda1
#     # except:
#     #     pass
#     print(a, soc24mathlib.factor(a))
#     assert(soc24mathlib.factor(a) == soc24mathlib.probabilistic_factor(a))

import soc24mathlib
import random

for i in range(1000000):
    if i % 2 == 0:
        p = soc24mathlib.gen_prime(1000000000)
        # print(p)
        assert(soc24mathlib.aks_test(p) == True)
    else:
        p = random.randint(1000000000, 1000000000000)
        assert(soc24mathlib.aks_test(p) == soc24mathlib.is_prime(p))
        # print(p)
    if i % 100 == 0:
        print(i)