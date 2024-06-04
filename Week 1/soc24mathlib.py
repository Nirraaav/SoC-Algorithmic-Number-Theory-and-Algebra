def pair_gcd(a: int, b: int) -> int:
    if (a < b):
        temp = a
        a = b
        b = temp

    while b != 0:
        r = a % b
        a = b
        b = r

    return a
    

# def pair_egcd(a: int, b: int) -> tuple[int, int, int]:

# def gcd(*args: int) -> int:

def pair_lcm(a: int, b: int) -> int:
    g = pair_gcd(a, b)
    l = (a * b) / g
    return l

# def lcm(*args: int) -> int:

# def are_relatively_prime(a: int, b: int) -> bool:

# def mod_inv(a: int, n: int) -> int:

# def crt(a: list[int], n: list[int]) -> int:

def pow(a: int, m: int, n: int) -> int:
    res = 1
    a %= n
    if (a == 0 and m == 0):
        return 1
    if (a == 0):
        return 0
    while (m > 0):
        if(m % 2 == 1):
            res = (res * a) % n
        a = (a * a) % n
        m //= 2
    return res % n

# def is_quadratic_residue_prime(a: int, p: int) -> int:

# def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:






