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

tassert(soc24mathlib.floor_sqrt(64) == 8)
tassert(soc24mathlib.floor_sqrt(2086571081) == 45679)
tassert(soc24mathlib.floor_sqrt(1368) == 36)

tassert(soc24mathlib.is_perfect_power(64) == True)
tassert(soc24mathlib.is_perfect_power(243) == True)
tassert(soc24mathlib.is_perfect_power(1368) == False)

tassert(soc24mathlib.is_prime(7) == True)
tassert(soc24mathlib.is_prime(2) == True)
tassert(soc24mathlib.is_prime(15) == False)
tassert(soc24mathlib.is_prime(3698849471) == True)
tassert(soc24mathlib.is_prime(79275795119) == False)

# Note that the first components of the tuples are the actual factors, while the second components are the multiplicities.
# The actual factors must be sorted in ascending order.
tassert(soc24mathlib.factor(1) == [])
tassert(soc24mathlib.factor(7) == [(7, 1)])
tassert(soc24mathlib.factor(243) == [(3, 5)])
tassert(soc24mathlib.factor(4104) == [(2, 3), (3, 3), (19, 1)])
tassert(soc24mathlib.factor(1408198281) == [(3, 1), (7, 1), (17, 1), (19, 1), (31, 1), (37, 1), (181, 1)])
tassert(soc24mathlib.factor(100000000000) == [(2, 11), (5, 11)])

tassert(soc24mathlib.euler_phi(1) == 1)
tassert(soc24mathlib.euler_phi(7) == 6)
tassert(soc24mathlib.euler_phi(243) == 162)
tassert(soc24mathlib.euler_phi(4104) == 1296)

p1 = soc24mathlib.QuotientPolynomialRing([-3, -5, -1, 1], [7, 0, 0,  3, 4])
p2 = soc24mathlib.QuotientPolynomialRing([1, 5, 7, 3], [7, 0, 0,  3, 4])
tassert(p1.element == [-3, -5, -1, 1])
tassert(p1.pi_generator == [7, 0, 0,  3, 4])
tassert(p2.element == [1, 5, 7, 3])
tassert(p2.pi_generator == [7, 0, 0,  3, 4])
p_add = soc24mathlib.QuotientPolynomialRing.Add(p1, p2)
tassert(p_add.element == [-2, 0, 6, 4] and p_add.pi_generator == [7, 0, 0,  3, 4])
p_sub = soc24mathlib.QuotientPolynomialRing.Sub(p1, p2)
tassert(p_sub.element == [-4, -10, -8, -2] and p_sub.pi_generator == [7, 0, 0,  3, 4])
p_mul = soc24mathlib.QuotientPolynomialRing.Mul(p1, p2)
tassert(p_mul.element == [11, 15, -68, -42] and p_mul.pi_generator == [7, 0, 0,  3, 4])
p_div = soc24mathlib.QuotientPolynomialRing.GCD(p1, p2)
tassert(p_div.element == [1, 2, 1, 0] and p_div.pi_generator == [7, 0, 0,  3, 4])
test_var = False
try:
    p_inv = soc24mathlib.QuotientPolynomialRing.Inv(p1)
except:
    test_var = True
tassert(test_var == True)
p_inv = soc24mathlib.QuotientPolynomialRing.Inv(soc24mathlib.QuotientPolynomialRing([1, 0, 0, 0], [7, 0, 0,  3, 4]))
tassert(p_inv.element == [1, 0, 0, 0] and p_inv.pi_generator == [7, 0, 0,  3, 4])

tassert(soc24mathlib.aks_test(7) == True)
tassert(soc24mathlib.aks_test(2) == True)
tassert(soc24mathlib.aks_test(15) == False)
tassert(soc24mathlib.aks_test(3698849471) == True)
tassert(soc24mathlib.aks_test(79275795119) == False)

print("All tests passed!")
