import soc24mathlib
import random

for i in range(100):
    if i % 2 == 0:
        p = soc24mathlib.gen_prime(100000000000000000000000)
        assert(soc24mathlib.aks_test(p) == True)
    else:
        p = random.randint(10000000000000000000, 10000000000000000000000)
        assert(soc24mathlib.aks_test(p) == soc24mathlib.is_prime(p))

    if i % 10 == 0:
        print(i)
    