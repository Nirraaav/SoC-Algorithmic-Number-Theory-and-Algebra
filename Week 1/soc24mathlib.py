def pair_gcd(a: int, b: int) -> int:
    """
    Returns the greatest common divisor of two integers a and b.

    Parameters:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The greatest common divisor of a and b.
    """

    if (a < b):
        temp = a
        a = b
        b = temp

    while b != 0:
        r = a % b
        a = b
        b = r

    return a
    
def pair_egcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Returns the greatest common divisor of two integers a and b, and the coefficients of Bezout's identity.

    Parameters:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        tuple[int, int, int]: A tuple containing the greatest common divisor of a and b, and the coefficients of Bezout's identity.
    """
    swap = False
    if (a < b):
        temp = a
        a = b
        b = temp
        swap = True

    s, s_dash, t, t_dash = 1, 0, 0, 1
    while b != 0:
        q = a // b
        r = a % b
        a, s, t, b, s_dash, t_dash = b, s_dash, t_dash, r, s - s_dash * q, t - t_dash * q

    if swap:
        return t, s, a
    return s, t, a    

def gcd(*args: int) -> int:
    """
    Returns the greatest common divisor of a list of integers.

    Parameters:
        args (int): A list of integers.

    Returns:
        int: The greatest common divisor of the list of integers.
    """
    
    # if (len(args) == 0):
    #     return 0
    # if (len(args) == 1):
    #     return args[0]
    if (len(args) == 2):
        return pair_gcd(args[0], args[1])

    g = pair_gcd(args[0], args[1])
    for i in range(2, len(args)):
        g = pair_gcd(g, args[i])

    return g

def pair_lcm(a: int, b: int) -> int:
    """
    Returns the least common multiple of two integers a and b.

    Parameters:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The least common multiple of a and b.
    """
    g = pair_gcd(a, b)
    # return exact value instead of scientific notation
    return (a * b) // g

def lcm(*args: int) -> int:
    """
    Returns the least common multiple of a list of integers.
    
    Parameters:
        args (int): A list of integers.

    Returns:
        int: The least common multiple of the list of integers.
    """
    # if (len(args) == 0):
    #     return 0
    # if (len(args) == 1):
    #     return args[0]
    if (len(args) == 2):
        return pair_lcm(args[0], args[1])

    l = pair_lcm(args[0], args[1])
    for i in range(2, len(args)):
        l = pair_lcm(l, args[i])

    return l

def are_relatively_prime(a: int, b: int) -> bool:
    """
    Returns True if two integers a and b are relatively prime, and False otherwise.

    Parameters:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        bool: True if a and b are relatively prime, and False otherwise.
    """
    return pair_gcd(a, b) == 1

def mod_inv(a: int, n: int) -> int:
    """
    Returns the modular inverse of an integer a modulo n.

    Parameters:
        a (int): The integer whose modular inverse is to be found.
        n (int): The modulo.

    Returns:
        int: The modular inverse of a modulo n.

    Raises:
        Exception: If a and n are not relatively prime.
    """
    if pair_gcd(a, n) != 1:
        raise Exception("The numbers are not relatively prime.")
    s, t, d = pair_egcd(a, n)
    return s % n

def crt(a: list[int], n: list[int]) -> int:
    """
    Returns the solution to a system of linear congruences using the Chinese Remainder Theorem.

    Parameters:
        a (list[int]): A list of integers representing the remainders.
        n (list[int]): A list of integers representing the moduli.

    Returns:
        int: The solution to the system of linear congruences.
    """
    N = 1
    for i in range(len(n)):
        N *= n[i]

    x = 0
    for i in range(len(n)):
        N_i = N // n[i]
        x += a[i] * N_i * mod_inv(N_i, n[i])
    return x % N

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

def pow_without_mod(a: int, m: int) -> int:
    """
    Returns a raised to the power m.

    Parameters:
        a (int): The base.
        m (int): The exponent.

    Returns:
        int: a raised to the power m.
    """
    res = 1
    if (a == 0 and m == 0):
        return 1
    if (a == 0):
        return 0
    while (m > 0):
        if(m % 2 == 1):
            res = res * a
        a = a * a
        m //= 2
    return res

def is_quadratic_residue_prime(a: int, p: int) -> int:
    """
    Returns 1 if a is a quadratic residue modulo p, -1 if a is a quadratic non-residue modulo p and 0 if a is not coprime to p using Euler's criterion.

    Parameters:
        a (int): The integer whose quadratic residue is to be determined.
        p (int): The modulo. (prime)

    Returns:
        int: 1 if a is a quadratic residue modulo p, -1 if a is a quadratic non-residue modulo p and 0 if a is not coprime to p.
    """
    if pair_gcd(a, p) != 1:
        return 0
    if pow(a, (p - 1) // 2, p) == 1:
        return 1
    if pow(a, (p - 1) // 2, p) == p - 1 or pow(a, (p - 1) // 2, p) == -1:
        return -1

def phi(n: int) -> int:
    """
    Returns the Euler's totient function of n.

    Parameters:
        n (int): The integer whose Euler's totient function is to be determined.

    Returns:
        int: The Euler's totient function of n.
    """
    # count = 0
    # for i in range(1, n):
    #     if pair_gcd(i, n) == 1:
    #         count += 1
    # return count
    phi = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            phi -= phi // p
        p += 1
    if n > 1:
        phi -= phi // n

    return phi


def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    """
    Returns 1 if a is a quadratic residue modulo p^e, -1 if a is a quadratic non-residue modulo p^e and 0 if a is not coprime to p.

    Parameters:
        a (int): The integer whose quadratic residue is to be determined.
        p (int): The prime.
        e (int): The power.

    Returns:
        int: 1 if a is a quadratic residue modulo p^e, -1 if a is a quadratic non-residue modulo p^e and 0 if a is not coprime to p.
    """
    # prime_pow = pow_without_mod(p, e)
    # if pair_gcd(a, p) != 1:
    #     return 0
    # if pair_gcd(a, prime_pow) != 1:
    #     return 0
    # if e == 1:
    #     return is_quadratic_residue_prime(a, p)
    # if pow(a, phi(prime_pow)//2, prime_pow) == 1:
    #     return 1
    # if pow(a, phi(prime_pow)//2, prime_pow) == prime_pow - 1 or pow(a, phi(prime_pow)//2, prime_pow) == -1:
    #     return -1
    # return 0

    if pair_gcd(a, p) != 1:
        return 0
    # if e == 1:
    #     return is_quadratic_residue_prime(a, p)
    # if is_quadratic_residue_prime(a, p) == 0:
    #     return 0
    # if is_quadratic_residue_prime(a, p) == 1:
    #     return 1
    # if is_quadratic_residue_prime(a, p) == -1:
    #     return -1
    return is_quadratic_residue_prime(a, p)
    
# print(is_quadratic_residue_prime(3, 7))