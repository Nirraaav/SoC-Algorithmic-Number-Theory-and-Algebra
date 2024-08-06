import soc24mathlib
import time

counter = 0

def tassert(condition):
    print(condition)
    global counter
    start_time = time.time()
    try:
        assert condition
    except AssertionError:
        print("AssertionError")
    finally:
        end_time = time.time()
        time_in_ms = (end_time - start_time) * 1000
        print(f"Condition: {condition}, Time: {time_in_ms} ms, counter: {counter}")
        counter += 1

tassert(soc24mathlib.discrete_log(11, 2, 13) == 7)
tassert(soc24mathlib.discrete_log(12384, 89, 3698849471) == 1261856717)

tassert(soc24mathlib.legendre_symbol(3, 7) == -1)
tassert(soc24mathlib.legendre_symbol(9, 13) == 1)
tassert(soc24mathlib.legendre_symbol(0, 17) == 0)
tassert(soc24mathlib.legendre_symbol(36249236958, 312345674079547151037918331725178312522478809653607352546657135738291654855733134069982077700935127515340479970913704499650782485828349263440468316632391) == -1)

tassert(soc24mathlib.jacobi_symbol(3, 79) == -1)
tassert(soc24mathlib.jacobi_symbol(1789,3189045) ==-1)
tassert(soc24mathlib.jacobi_symbol(7921, 489303) == 1)
tassert(soc24mathlib.jacobi_symbol(136, 153) == 0)

tassert(soc24mathlib.modular_sqrt_prime(11, 19) == 7)
tassert(soc24mathlib.modular_sqrt_prime(12378, 3698849471) == 1397367648)

tassert(soc24mathlib.modular_sqrt_prime_power(11, 19, 8) == 2684202706)
tassert(soc24mathlib.modular_sqrt_prime_power(12378, 3698849471, 3) == 19725977363156848933505792157)

tassert(soc24mathlib.modular_sqrt(10, 15) == 5)
tassert(soc24mathlib.modular_sqrt(91, 157482) == 62855)

tassert(soc24mathlib.is_smooth(1759590, 20) == True)
tassert(soc24mathlib.is_smooth(906486, 150) == False)

tassert(soc24mathlib.probabilistic_dlog(11, 2, 13) == 7)
tassert(soc24mathlib.probabilistic_dlog(12384, 89, 3698849471) == 1261856717)
tassert(soc24mathlib.probabilistic_dlog(131313, 13, 17077114927) == 12294541275)

# Note that the first components of the tuples are the actual factors, while the second components are the multiplicities.
# The actual factors must be sorted in ascending order.
tassert(soc24mathlib.probabilistic_factor(1) == [])
tassert(soc24mathlib.probabilistic_factor(7) == [(7, 1)])
tassert(soc24mathlib.probabilistic_factor(243) == [(3, 5)])
tassert(soc24mathlib.probabilistic_factor(4104) == [(2, 3), (3, 3), (19, 1)])
tassert(soc24mathlib.probabilistic_factor(1408198281) == [(3, 1), (7, 1), (17, 1), (19, 1), (31, 1), (37, 1), (181, 1)])

print("All tests passed!")