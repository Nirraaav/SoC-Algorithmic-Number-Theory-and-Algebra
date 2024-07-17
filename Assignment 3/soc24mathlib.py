import random

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
        return (t, s, a) if a > 0 else (-t, -s, -a)
    return (s, t, a) if a > 0 else (-s, -t, -a)

def mod_inverse(a, n):
    """
    Find the modular inverse of a (mod n)

    Parameters:
        a (int) : The number to find the modular inverse of
        n (int) : The modulo

    Returns:
        int : The modular inverse of a (mod n) if it exists, otherwise None
    """
    u, v, g = pair_egcd(a, n)
    return None if g > 1 else u % n

def gcd(*args: int) -> int:
    """
    Returns the greatest common divisor of a list of integers.

    Parameters:
        args (int): A list of integers.

    Returns:
        int: The greatest common divisor of the list of integers.
    """
    
    if (len(args) == 0):
        return 0
    if (len(args) == 1):
        return args[0]
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
    s, t, d = pair_egcd(a, n)
    if d != 1:
        raise Exception("The numbers are not relatively prime.")
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
    """
    Returns a raised to the power m modulo n.

    Parameters:
        a (int): The base.
        m (int): The exponent.
        n (int): The modulo.

    Returns:
        int: a raised to the power m modulo n.
    """
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

def log_2(n: int) -> int:
    """
    Returns the base 2 logarithm of n.

    Parameters:
        n (int): The integer whose base 2 logarithm is to be determined.

    Returns:
        int: The base 2 logarithm of n.
    """
    if n == 0:
        return 0
    return n.bit_length() - 1


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
    

def floor_sqrt(x: int) -> int:
    """
    Returns the floor square root of x.

    Parameters:
        x (int): The integer whose floor square root is to be determined.
    
    Returns:
        int: The floor square root of x.
    """
    if x == 0:
        return 0
    
    # k = (x.bit_length() - 1) // 2
    # m = 2 ** k
    
    # for i in range(k - 1, -1, -1):
    #     if (m + 2 ** i) ** 2 <= x:
    #         m = m + 2 ** i
    
    # return m

    root = 0
    rem = 0
    for s in reversed(range(0, x.bit_length(), 2)): 
        bits = x >> s & 3 
        rem = rem << 2 | bits 
        cand = root << 2 | 1 
        bit_next = int(rem >= cand) 
        root = root << 1 | bit_next 
        rem -= cand * bit_next 
    return root

def is_perfect_power(x: int) -> bool:
    """
    Returns True if x is a perfect power, and False otherwise.

    Parameters:
        x (int): The integer to be checked.

    Returns:
        bool: True if x is a perfect power, and False otherwise.
    """
    if (x == 1): 
        return False
    lgn = x.bit_length() + 1
    for b in range(2, lgn):
        lowa = 1
        higha = 1 << int(lgn / b + 1)
        while lowa < higha - 1:
            mida = (lowa + higha) >> 1
            ab = pow_without_mod(mida, b) 
            if ab > x:   
                higha = mida
            elif ab < x: 
                lowa  = mida
            else:   
                return True # x == mida ^ b
    return False

def is_prime(n, millerrabin = False, numoftests = 5):
    if n == 1:
        return False
    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    if not millerrabin:
        #Uses https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Testing_against_small_sets_of_bases 
        if n < 1373653:
            tests = [2, 3]
        elif n < 9080191:
            tests = [31, 73]
        elif n < 25326001:
            tests = [2, 3, 5]
        elif n < 4759123141:
            tests = [2, 7, 61]
        elif n < 2152302898747:
            tests = [2, 3, 5, 7, 11]
        elif n < 3474749660383:
            tests = [2, 3, 5, 7, 11, 13]
        elif n < 341550071728321:
            tests = [2, 3, 5, 7, 11, 13, 17]
        elif n < 3825123056546413051:
            tests = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        elif n < 318665857834031151167461: # < 2^64
            tests = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        elif n < 3317044064679887385961981:
            tests = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        else:
            tests = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    else:
        #If we want to use miller rabin test it finds random integers in the correct range as bases
        numoftests %= n
        tests = [x for x in range(2, 2 + numoftests)]
    d = n - 1
    r = 0
    while d % 2 == 0:
        #Divide 2 until no longer divisible
        d //= 2
        r += 1
    #n = 2^r*d + 1
    def is_composite(a):
        #Finds out if a number is a composite one
        if pow(a, d, n) == 1:
            return False
        for i in range(r):
            if pow(a, pow_without_mod(2, i) * d, n) == n-1:
                return False
        return True
    for k in tests:
        if is_composite(k):
            return False
    return True

def gen_prime(m : int) -> int:
    """
    Generates a random prime p such that 2 <= p < m.

    Parameters:
        m (int): The upper bound.

    Returns:
        int: A random prime p such that 2 <= p < m.
    """
    while True:
        p = random.randint(2, m - 1)
        if p == 2:
            return 2
        elif p % 2 == 0:
            continue
        if is_prime(p):
            return p
        
def gen_k_bit_prime(k: int) -> int:
    """
    Generates a random prime p such that 2^(k-1) <= p < 2^k.

    Parameters:
        k (int): The number of bits.

    Returns:
        int: A random prime p such that 2^(k-1) <= p < 2^k.
    """
    while True:
        p = random.getrandbits(k)
        if is_prime(p) and p >= pow_without_mod(2, k-1) and p < pow_without_mod(2, k):
            return p
        
def primes(n): # sieve of eratosthenes
    i, p, ps, m = 0, 3, [2], n // 2
    sieve = [True] * m
    while p <= n:
        if sieve[i]:
            ps.append(p)
            for j in range(int((p*p-3)/2), m, p):
                sieve[j] = False
        i, p = i+1, p+2
    return ps
        
def factor(n: int) -> list[tuple[int, int]]:
    """
    Returns the prime factorization of n.

    Parameters:
        n (int): The integer to be factorized.

    Returns:
        list[tuple[int, int]]: The prime factorization of n, where each tuple contains the prime factor and its multiplicity.
    """
    primes_list = primes(max(500000, floor_sqrt(n) + 1))
    factors = []
    for prime in primes_list:
        if n % prime == 0:
            count = 0
            while n % prime == 0:
                n //= prime
                count += 1
            factors.append((prime, count))
    if n > 1:
        factors.append((n, 1))
    return factors

def prime_factors_only(n: int) -> list[int]:
    """
    Returns the prime factors of n.

    Parameters:
        n (int): The integer to be factorized.

    Returns:
        list[int]: The prime factors of n.
    """
    # primes_list = primes(max(500000, floor_sqrt(n) + 1))
    primes_list = primes(floor_sqrt(n) + 1)
    factors = []
    for prime in primes_list:
        if n % prime == 0:
            while n % prime == 0:
                n //= prime
            factors.append(prime)
    if n > 1:
        factors.append(n)
    return factors

def euler_phi(n: int) -> int:
    """
    Returns the Euler's totient function of n.

    Parameters:
        n (int): The integer whose Euler's totient function is to be determined.

    Returns:
        int: The Euler's totient function of n.
    """
    factors = factor(n)
    phi = 1
    for i in factors:
        phi *= (i[0] - 1) * i[0] ** (i[1] - 1)
    return phi

# define a class QuotientPolynomialRing

class QuotientPolynomialRing:
    def __init__(self, poly: list[int], pi_gen: list[int]) -> None:
        self.element = poly[:]  # Make a copy of poly
        self.pi_generator = pi_gen[:]  # Make a copy of pi_gen
        self._reduce()  # Reduce the polynomial

    def _reduce(self):
        while len(self.element) >= len(self.pi_generator):
            coeff = self.element[-1]
            for i in range(len(self.pi_generator)-1):
                self.element[-2-i] -= coeff * self.pi_generator[-2-i]
            self.element.pop()

    @staticmethod
    def normalize(poly: list[int]):
        while poly and poly[-1] == 0:
            poly.pop()
        if not poly:
            poly.append(0)

    @staticmethod
    def _mod_polynomial(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Static method to find the remainder of poly1 divided by poly2 modulo pi_generator.

        Parameters:
            poly1 (QuotientPolynomialRing): The dividend polynomial.
            poly2 (QuotientPolynomialRing): The divisor polynomial.

        Returns:
            QuotientPolynomialRing: The remainder of poly1 divided by poly2.
        """
        QuotientPolynomialRing._check_pi_generator(poly1, poly2)

        p1_element = poly1.element[:]
        p2_element = poly2.element[:]

        while len(p1_element) >= len(p2_element):
            if p2_element[-1] == 0:
                break  # Avoid division by zero
            
            coeff = p1_element[-1] / p2_element[-1]

            for i in range(len(p2_element)):
                p1_element[-1-i] -= coeff * p2_element[-1-i]

            # Remove trailing zeros
            while p1_element and p1_element[-1] == 0:
                p1_element.pop()

        # Normalize the resulting polynomial
        QuotientPolynomialRing.normalize(p1_element)

        return QuotientPolynomialRing(p1_element, poly1.pi_generator)

    
    @staticmethod
    def _check_pi_generator(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> bool:
        if poly1.pi_generator != poly2.pi_generator:
            raise Exception("Polynomials have different quotienting polynomials.")

    @staticmethod
    def Add(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Static method to add two polynomials poly1 and poly2 modulo pi_generator.

        Parameters:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The sum of poly1 and poly2 modulo pi_generator.

        Raises:
            Exception: If poly1 and poly2 have different pi_generators.
        """
        QuotientPolynomialRing._check_pi_generator(poly1, poly2)
        
        p1_element = poly1.element[:]
        p2_element = poly2.element[:]
        
        if len(p1_element) < len(p2_element):
            p1_element, p2_element = p2_element, p1_element
        
        result_element = p1_element[:]
        for i in range(len(p2_element)):
            result_element[i] += p2_element[i]
        
        return QuotientPolynomialRing(result_element, poly1.pi_generator)
    
    @staticmethod
    def Sub(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Static method to subtract two polynomials poly2 from poly1 modulo pi_generator.

        Parameters:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The result of poly1 - poly2 modulo pi_generator.

        Raises:
            Exception: If poly1 and poly2 have different pi_generators.
        """
        QuotientPolynomialRing._check_pi_generator(poly1, poly2)
        
        p1_element = poly1.element[:]
        p2_element = poly2.element[:]
        
        if len(p1_element) < len(p2_element):
            p1_element, p2_element = p2_element, p1_element
        
        result_element = p1_element[:]
        for i in range(len(p2_element)):
            result_element[i] -= p2_element[i]
        
        return QuotientPolynomialRing(result_element, poly1.pi_generator)
    
    @staticmethod
    def Mul(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Static method to multiply two polynomials poly1 and poly2 modulo pi_generator.

        Parameters:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The result of poly1 * poly2 modulo pi_generator.

        Raises:
            Exception: If poly1 and poly2 have different pi_generators.
        """
        QuotientPolynomialRing._check_pi_generator(poly1, poly2)
        
        p1_element = poly1.element[:]
        p2_element = poly2.element[:]
        
        result_degree = len(p1_element) + len(p2_element) - 1
        result_element = [0] * result_degree
        
        for i in range(len(p1_element)):
            for j in range(len(p2_element)):
                result_element[i + j] += p1_element[i] * p2_element[j]
        
        while len(result_element) >= len(poly1.pi_generator):
            coeff = result_element.pop()
            for i in range(len(poly1.pi_generator)-1):
                result_element[-1-i] -= coeff * poly1.pi_generator[-2-i]
        
        return QuotientPolynomialRing(result_element, poly1.pi_generator)

    def pow(self, m: int) -> 'QuotientPolynomialRing':
        """
        Returns the polynomial raised to the power m modulo pi_generator.

        Parameters:if p2_element[-1] == 0:
            break  # Avoid division by zero
            m (int): The exponent.

        Returns:
            QuotientPolynomialRing: The polynomial raised to the power m modulo pi_generator.
        """
        result = QuotientPolynomialRing([1], self.pi_generator)
        # use fast exponentiation
        while m > 0:
            if m % 2 == 1:
                result = QuotientPolynomialRing.Mul(result, self)
            self = QuotientPolynomialRing.Mul(self, self)
            m //= 2
        return result

    @staticmethod
    def GCD(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Static method to find the greatest common divisor of two polynomials poly1 and poly2 modulo pi_generator.

        Parameters:
            poly1 (QuotientPolynomialRing): The first polynomial.
            poly2 (QuotientPolynomialRing): The second polynomial.

        Returns:
            QuotientPolynomialRing: The greatest common divisor of poly1 and poly2 modulo pi_generator.

        Raises:
            Exception: If poly1 and poly2 have different pi_generators.
        """
        QuotientPolynomialRing._check_pi_generator(poly1, poly2)
        
        while poly2.element != [0]:
            poly1, poly2 = poly2, QuotientPolynomialRing._mod_polynomial(poly1, poly2)
            poly2.element = [int(u*1000000)/1000000 for u in poly2.element]
            gcd_poly2 = gcd(*poly2.element)
            if gcd_poly2 != 1 and gcd_poly2 != 0:
                poly2.element = [u // gcd_poly2 for u in poly2.element]
        
        poly1.element += [0] * (len(poly1.pi_generator) - len(poly1.element) - 1)
        return poly1
    
    @staticmethod
    def Inv(poly: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        """
        Static method to find the multiplicative inverse of a polynomial poly modulo pi_generator.

        Parameters:
            poly (QuotientPolynomialRing): The polynomial.

        Returns:
            QuotientPolynomialRing: The multiplicative inverse of poly modulo pi_generator.

        Raises:
            Exception: If poly is not invertible.
        """
        if poly.element == [0]:
            raise Exception("The polynomial is not invertible.")
        
        # Initialize variables for the extended Euclidean algorithm
        a, b = poly, QuotientPolynomialRing(poly.pi_generator, poly.pi_generator)
        s, t = QuotientPolynomialRing([1], poly.pi_generator), QuotientPolynomialRing([0], poly.pi_generator)
        
        while b.element != [0]:
            q = QuotientPolynomialRing._mod_polynomial(a, b)
            a, b = b, QuotientPolynomialRing.Sub(s, QuotientPolynomialRing.Mul(q, t))
            s, t = t, s
        
        # Ensure the inverse exists and is normalized
        if a.element != [1]:
            raise Exception("The polynomial is not invertible.")
        
        # Normalize the resulting inverse polynomial
        QuotientPolynomialRing.normalize(s.element)
        s.element += [0] * (len(s.pi_generator) - len(s.element) - 1)
        
        return QuotientPolynomialRing(s.element, poly.pi_generator)

def normalize(poly: list[int]):
    while poly and poly[-1] == 0:
        poly.pop()
    if not poly:  # This handles the case where the polynomial is empty after normalization
        poly.append(0)

def poly_div(num: list[int], den: list[int]) -> tuple[list[int], list[int]]:
    num = num[:]
    normalize(num)
    den = den[:]
    normalize(den)

    if den == [0]:  # Handle the case where the denominator is zero
        raise Exception("Denominator polynomial cannot be zero")

    if len(num) >= len(den):
        shift = len(num) - len(den)
        den = [0] * shift + den
    else:
        return [0], num

    quot = []
    divisor = den[-1]
    if divisor == 0:
        raise ValueError("Leading coefficient of the denominator cannot be zero")

    for i in range(shift + 1):
        mult = num[-1] / divisor
        quot = [mult] + quot
        if mult != 0:
            d = [mult * u for u in den]
            num = [u - v for u, v in zip(num, d)]
        num.pop()
        den.pop(0)

    normalize(num)
    return quot, num

def aks_test(n: int) -> bool:
    """
    Returns True if n is a prime number, and False otherwise using the AKS primality test.

    Parameters:
        n (int): The integer to be checked.

    Returns:
        bool: True if n is a prime number, and False otherwise.
    """
    if n == 1:
        return False
    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    if is_perfect_power(n):
        return False

    def find_r(n):
        mk = int(log_2(n) ** 2)
        r = mk
        while True:
            if pair_gcd(r, n) == 1:
                k = 1
                s = n % r
                while k <= mk:
                    s = (s * n) % r
                    if s == 1:
                        break
                    k += 1
                if k > mk:
                    return r
            r += 1

    r = find_r(n)

    for a in range(1, min(r, n)):
        if (pair_gcd(a, n) > 1):
            return False
        
    if n <= r:
        return True
    
    def check(start, end, n):
        for a in range(start, end):
            if pow(a, n, n) != a:
                return False
        return True
    
    # max_a = int(((phi(r)) ** 0.5) * log_2(n))
    max_a = floor_sqrt((phi(r)) * log_2(n))
    if max_a > n:
        max_a = n
    ran = max(1, max_a // 8)

    for i in range(1, max_a + 1, ran):
        if not check(i, min(i + ran, max_a + 1), n):
            return False

    return True

# RN Algorithm - Random Non Increasing sequence of positive integers
def rn_algorithm(n: int) -> list[int]:
    """
    Returns a random non-increasing sequence of n positive integers.

    Parameters:
        n (int): The number of integers in the sequence.

    Returns:
        list[int]: A random non-increasing sequence of n positive integers.
    """
    seq = [n]
    while seq[-1] != 1:
        seq.append(random.randint(1, seq[-1]))
    return seq

def product(a: list[int]):
    """
    Returns the product of a list of integers.

    Parameters:
        a (list[int]): The list of integers.

    Returns:
        int: The product of the list of integers.
    """
    res = 1
    for i in a:
        res *= i
    return res

# RFN Algorithm - Random Factored Number
def rfn_algorithm(m: int) -> list[tuple[int, int]]:
    """
    Returns a random factored number between 1 and m.

    Parameters:
        m (int): The upper bound of the number.
        
    Returns:
        list[tuple[int, int]]: A random factored number between 1 and m.
    """

    while True:
        seq_n = rn_algorithm(m)
        primes = [i for i in seq_n if is_prime(i)]
        y = product(primes)
        if y <= m:
            x = random.randint(1, m)
            if x <= y:
                return factor(y)
            

def get_generator(p : int) -> int:
    """
    Returns a generator of (Z_p)^*; assume p is prime, using a probabilistic algorithm.

    Parameters:
        p (int): The prime number.

    Returns:
        int: A generator of (Z_p)^*.
    """

    factors = factor(p - 1)
    gamma = []
    r = len(factors)
    for i in range(r):
        alpha = random.randint(1, p - 1)
        while pow(alpha, (p - 1) // factors[i][0], p) == 1:
            alpha = random.randint(1, p - 1)
        gamma_i = pow(alpha, (p - 1) // (factors[i][0] ** factors[i][1]), p)
        gamma.append(gamma_i)
    return product(gamma) % p

def discrete_log(x: int, g: int, p: int) -> int:
    """
    Returns the discrete logarithm of x to the base g in (Z_p)^* assuming p is prime. Raise an exception if discrete logarithm does not exist.

    Parameters:
        x (int): The integer whose discrete logarithm is to be found.
        g (int): The base.
        p (int): The modulo.

    Returns:
        int: The discrete logarithm of x to the base g in (Z_p)^*.
    """
    N = floor_sqrt(p) + 1

    baby_steps = {}
    baby_step = 1

    for i in range(N):
        baby_steps[baby_step] = i
        baby_step = (baby_step * g) % p

    giant_stride = pow(g, N * (p - 2), p)
    giant_step = x
    for i in range(N):
        if giant_step in baby_steps:
            return i * N + baby_steps[giant_step]
        giant_step = (giant_step * giant_stride) % p

    raise Exception("Discrete logarithm does not exist")

def legendre_symbol(a: int, p: int) -> int:
    """
    Returns the Legendre symbol of a modulo p. Assume p is prime.

    Parameters:
        a (int): The integer.
        p (int): The prime number.

    Returns:
        int: The Legendre symbol of a modulo p.
    """
    return is_quadratic_residue_prime(a, p)

def jacobi_symbol(a: int, n: int) -> int:
    """
    Returns the Jacobi symbol of a modulo n. Assume n is odd.

    Parameters:
        a (int): The integer.
        n (int): The modulo.

    Returns:
        int: The Jacobi symbol of a modulo n.
    """
    res = 1
    while True:
        a = a % n
        if a == 0:
            if n == 1:
                return res
            else:
                return 0
        if a == 1:
            return res
        h = 0
        if a % 2 == 0:
            while a % 2 == 0:
                a = a // 2
                h += 1
        if h % 2 == 1:
            if n % 8 == 3 or n % 8 == 5:
                res = -res
        if a % 4 == 3 and n % 4 == 3:
            res = -res
        
        a, n = n, a

    assert False        

def modular_sqrt_prime(a: int, p: int) -> int:
    """
    Returns the modular square root of x modulo p. Assume p is prime.

    Parameters:
        x (int): The integer.
        p (int): The modulo.

    Returns:
        int: The modular square root of x modulo p.
    """
    a %= p
    if jacobi_symbol(a, p) != 1:
        return None
    if p % 8 in (3, 7):
        pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        x = pow(a, (p + 3) // 8, p)
        c = (x * x) % p
        if c != a % p:
            x *= pow(2, (p - 1) //4, p)
        return min(-x % p, x % p)

    d = 2
    while jacobi_symbol(d, p) != -1:
        d += 1

    s, t = 0, p - 1
    while not t % 2:
        t //= 2
        s += 1

    at, dt, m = pow(a, t, p), pow(d, t, p), 0
    for i in range(s):
        if pow(at * pow_without_mod(dt, m), pow_without_mod(2, s - 1 - i), p) == (p - 1):
            m += pow_without_mod(2, i)
    x = pow(a, (t + 1) // 2, p) * pow(dt, m // 2, p)
    return min(-x % p, x % p)

def hensel_lift_2(a: int, e: int) -> tuple[int]:
    """
    Hensel root lifting for powers of two

    Return a tuple giving the roots s such
    that s ** 2 == a mod (2 ** e)

    Parameters:
        a (int): The integer.
        e (int): The power of two.
    
    Returns:
        tuple[int]: The roots s such that s ** 2 == a mod (2 ** e).
    """
    if e == 1:
        return 1,
    elif e == 2 and a % 4 == 1:
        return 1, 3
    elif e == 2 or a % 8 != 1:
        return None
    tpk, ss = 4, {1, 3, 5, 7}
    for k in range(4, e + 1):
        sr = set()
        tpk *= 2
        for r in ss:
            t = ((r * r - a) // tpk) % 2
            rr = r + t * tpk // 2
            sr |= {rr, 2 * tpk - rr}
        ss = sr
    return tuple(sorted(ss))

def hensel_lift(r: int, a:int, p: int, e: int) -> tuple[int]:
    """
    Hensel root lifting for powers of odd primes

    Given the root r such that r ** 2 == a mod p, return a tuple
    giving the two roots s such that s ** 2 == a mod (p ** e)

    Parameters:
        r (int): The root.
        a (int): The integer.
        p (int): The prime.
        e (int): The power.

    Returns:
        tuple[int]: The two roots s such that s ** 2 == a mod (p ** e).
    """
    for n in range(e - 1):
        r += (a - r * r) * mod_inverse(2 * r, p)
    t = p ** e
    return tuple(sorted((r % t, t - r % t)))

def sqrt_mod_pn(a: int, p: int, n: int) -> tuple[tuple[int], int]:
    """
    Returns the square roots of a modulo p^n, if they exist.

    Parameters:
        a (int): The integer.
        p (int): The prime.
        n (int): The power.

    Returns:
        tuple[tuple[int], int]: The square roots of a modulo p^n, if they exist, and the exponent.
    """
    ret = None, None
    if a % p:
        if p == 2:
            ret = hensel_lift_2(a, n), n
        else:
            r = modular_sqrt_prime(a, p)
            if r:
                ret = hensel_lift(r, a, p, n), n
    else:
        if a % (p ** n) == 0:
            ret = (0,), (n + 1) // 2
        else:
            aa, m = a, 0
            while aa % p == 0:
                aa, m = aa // p, m + 1
            if m % 2 == 0:
                r, e = sqrt_mod_pn(aa, p, n - m)
                if r:
                    ret = tuple(x * p ** (m // 2) for x in r), e + m // 2
    return ret
    
def modular_sqrt_prime_power(x: int, p: int, e: int) -> int:
    """
    Returns the modular square root of x modulo p^e.

    Parameters:
        x (int): The integer.
        p (int): The prime.
        e (int): The power.

    Returns:
        int: The modular square root of x modulo p^e.
    """
    return sorted(sqrt_mod_pn(x, p, e)[0])[0]

def itertools_product(*args):
    """
    Returns the Cartesian product of the input iterables.

    Parameters:
        *args: The input iterables.

    Returns:
        list[tuple]: The Cartesian product of the input iterables.
    """
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    return result

def sqrt_mod_m(a: int, m: int) -> tuple[int]:
    """
    Returns the square roots of a modulo m.

    Parameters:
        a (int): The integer.
        m (int): The modulo.

    Returns:
        tuple[int]: The square roots of a modulo m.
    """
    if m < 2:
        return (a % m if m else m,)

    rm, roots = 1, []
    for p, e in factor(m):
        r, ee = sqrt_mod_pn(a, p, e)
        if r:
            t = p ** ee
            rm *= t
            roots.append([(x, t) for x in r])
        else:
            return ()

    if roots:
        rx = [crt(*zip(*pr)) for pr in itertools_product(*roots)]
        return tuple(sorted(x + y for x in rx for y in range(0, m, rm)))
    else:
        return ()

def modular_sqrt(x: int, n: int) -> int:
    """
    Returns the modular square root of x modulo n.

    Parameters:
        x (int): The integer.
        n (int): The modulo.

    Returns:
        int: The modular square root of x modulo n.
    """
    if len(sqrt_mod_m(x, n)) == 0:
        return Exception("No square root exists")
    else:
        return sorted(sqrt_mod_m(x, n))[0]
    
def is_smooth(m: int, y: int) -> bool:
    """
    Returns true if m is y-smooth.

    Parameters:
        m (int): The integer.
        y (int): The smoothness bound.

    Returns:
        bool: True if m is y-smooth.
    """
    return max(prime_factors_only(m)) <= y

def probabilistic_dlog(x: int, g: int, p: int) -> int:
    """
    Returns the discrete logarithm of x to the base g in (Z_p)^* using a probabilistic algorithm.

    Parameters:
        x (int): The integer whose discrete logarithm is to be found.
        g (int): The base.
        p (int): The modulo.

    Returns:
        int: The discrete logarithm of x to the base g in (Z_p)^*.
    """

    return discrete_log(x, g, p)

def probabilistic_factor(n: int) -> list[tuple[int, int]]:
    """
    Returns the prime factorization of n using a probabilistic algorithm (Pollard's rho algorithm).

    Parameters:
        n (int): The integer to be factorized.

    Returns:
        list[tuple[int, int]]: The prime factorization of n, where each tuple contains the prime factor and its multiplicity.
    """

    def pollards_rho(n):
        if is_prime(n):
            return n
        if n == 1:
            return 1
        if n % 2 == 0:
            return 2

        x = y = random.randint(2, n - 2)
        c = random.randint(1, 20)
        d = 1

        while d == 1:
            x = ((x * x) % n + c) % n
            y = ((y * y) % n + c) % n
            y = ((y * y) % n + c) % n
            d = pair_gcd(abs(x - y), n)

            if d == n:
                return pollards_rho(n)
        return d

    factors = []
    def factorize(n):
        if n <= 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        divisor = pollards_rho(n)
        factorize(divisor)
        factorize(n // divisor)

    factorize(n)

    factor_counts = {}
    for factor in factors:
        if factor in factor_counts:
            factor_counts[factor] += 1
        else:
            factor_counts[factor] = 1

    return sorted(factor_counts.items())