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
    