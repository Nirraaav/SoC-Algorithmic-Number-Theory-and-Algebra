# Algorithmic Number Theory and Algebra 🔢

A pure Python library for computational number theory, providing a suite of functions for primality testing, factorization, modular arithmetic, polynomial operations, and more. This library is designed to be a self-contained toolkit for exploring concepts in number theory and algebra.

## ✨ Features

* **Modular Arithmetic**: GCD, LCM, Extended Euclidean Algorithm, Modular Inverse, Chinese Remainder Theorem.
* **Primality Testing**: Deterministic and probabilistic Miller-Rabin tests, and the AKS primality test.
* **Integer Factorization**: Trial division, and Pollard's Rho probabilistic algorithm.
* **Quadratic Residues**: Legendre and Jacobi symbols, and a robust modular square root solver using Tonelli-Shanks and Hensel's Lemma.
* **Discrete Logarithms**: Baby-step giant-step algorithm and primitive root finding.
* **Polynomials**: A powerful `QuotientPolynomialRing` class for arithmetic with polynomials over rational numbers.
* **Utility Functions**: Prime generation, Euler's totient function, perfect power checking, and more.

## 📜 Table of Contents

1.  [Basic Arithmetic & Modular Operations](#basic-arithmetic--modular-operations)
2.  [Primality Testing & Prime Generation](#primality-testing--prime-generation)
3.  [Integer Factorization](#integer-factorization)
4.  [Euler's Totient Function](#eulers-totient-function)
5.  [Modular Square Roots & Quadratic Residues](#modular-square-roots--quadratic-residues)
6.  [Discrete Logarithms & Group Generators](#discrete-logarithms--group-generators)
7.  [Polynomial Arithmetic](#polynomial-arithmetic)
8.  [Specialized & Utility Functions](#specialized--utility-functions)

---

## Basic Arithmetic & Modular Operations

Core functions for number-theoretic computations.

* `pair_gcd(a: int, b: int) -> int`
    * Computes the greatest common divisor (GCD) of two integers using the Euclidean algorithm.

* `gcd(*args: int) -> int`
    * Computes the GCD of a list of integers.

* `pair_egcd(a: int, b: int) -> tuple[int, int, int]`
    * Implements the Extended Euclidean Algorithm. Returns a tuple `(u, v, d)` such that $au + bv = d$, where $d = \text{gcd}(a, b)$.

* `mod_inverse(a, n) -> int`
    * Finds the modular multiplicative inverse of $a$ modulo $n$. Returns $x$ such that $ax \equiv 1 \pmod{n}$.

* `pair_lcm(a: int, b: int) -> int`
    * Computes the least common multiple (LCM) of two integers.

* `lcm(*args: int) -> int`
    * Computes the LCM of a list of integers.

* `are_relatively_prime(a: int, b: int) -> bool`
    * Checks if two integers are relatively prime (i.e., their GCD is 1).

* `crt(a: list[int], n: list[int]) -> int`
    * Solves a system of linear congruences using the Chinese Remainder Theorem. Finds an integer $x$ that satisfies $x \equiv a_i \pmod{n_i}$ for all $i$.

* `pow(a: int, m: int, n: int) -> int`
    * Calculates modular exponentiation $a^m \pmod{n}$ efficiently using the method of repeated squaring.

---

## Primality Testing & Prime Generation

Functions for identifying prime numbers and generating them.

* `is_prime(n, millerrabin=False, numoftests=5) -> bool`
    * A robust primality test. By default, it uses a deterministic set of bases for the Miller-Rabin test, which is guaranteed to be correct for numbers up to $3317 \times 10^{22}$. If `millerrabin` is `True`, it performs a probabilistic test with `numoftests` random bases.

* `aks_test(n: int) -> bool`
    * An implementation of the Agrawal-Kayal-Saxena (AKS) primality test. This is a deterministic polynomial-time algorithm for primality testing.

* `gen_prime(m: int) -> int`
    * Generates a random prime number $p$ such that $2 \le p < m$.

* `gen_k_bit_prime(k: int) -> int`
    * Generates a random prime number $p$ with exactly $k$ bits, i.e., $2^{k-1} \le p < 2^k$.

* `primes(n: int) -> list[int]`
    * Returns a list of all prime numbers up to $n$ using the Sieve of Eratosthenes.

---

## Integer Factorization

Functions to find the prime factors of an integer.

* `factor(n: int) -> list[tuple[int, int]]`
    * Returns the prime factorization of $n$ as a list of `(prime, exponent)` tuples. Uses trial division with a pre-computed list of primes up to $\sqrt{n}$.

* `probabilistic_factor(n: int) -> list[tuple[int, int]]`
    * Finds the prime factorization of $n$ using Pollard's rho algorithm, a probabilistic method that is more efficient for large numbers with small prime factors.

* `prime_factors_only(n: int) -> list[int]`
    * Returns a list of the unique prime factors of $n$.

---

## Euler's Totient Function

* `phi(n: int) -> int`
    * Calculates Euler's totient function, $\phi(n)$, which counts the number of positive integers up to $n$ that are relatively prime to $n$. This implementation uses prime factorization.

* `euler_phi(n: int) -> int`
    * An alternative implementation of Euler's totient function, also based on the prime factorization of $n$.

---

## Modular Square Roots & Quadratic Residues

A suite of functions for solving $x^2 \equiv a \pmod{n}$ and related problems.

* `legendre_symbol(a: int, p: int) -> int`
    * Computes the Legendre symbol $\left(\frac{a}{p}\right)$, which determines if $a$ is a quadratic residue modulo a prime $p$. Returns $1$ if it is a residue, $-1$ if not, and $0$ if $p|a$.

* `jacobi_symbol(a: int, n: int) -> int`
    * Computes the Jacobi symbol $\left(\frac{a}{n}\right)$, a generalization of the Legendre symbol for a composite, odd integer $n$.

* `modular_sqrt_prime(a: int, p: int) -> int`
    * Finds a solution to $x^2 \equiv a \pmod{p}$ where $p$ is prime. Implements the Tonelli-Shanks algorithm.

* `modular_sqrt_prime_power(x: int, p: int, e: int) -> int | None`
    * Finds a solution to $x^2 \equiv a \pmod{p^e}$ by lifting a solution from modulo $p$ using Hensel's Lemma.

* `modular_sqrt(x: int, n: int) -> int | Exception`
    * Finds the smallest non-negative integer solution to $x^2 \equiv a \pmod{n}$ for a composite modulus $n$. It combines solutions for each prime power factor of $n$ using the Chinese Remainder Theorem. Returns an exception if no solution exists.

---

## Discrete Logarithms & Group Generators 🔐

Functions for problems in finite cyclic groups.

* `get_generator(p: int) -> int`
    * Finds a generator (primitive root) of the multiplicative group of integers modulo $p$, $(\mathbb{Z}/p\mathbb{Z})^*$, where $p$ is a prime.

* `discrete_log(x: int, g: int, p: int) -> int`
    * Solves the discrete logarithm problem. Finds an integer $k$ such that $g^k \equiv x \pmod{p}$ using the baby-step giant-step algorithm.

* `probabilistic_dlog(x: int, g: int, p: int) -> int`
    * An alternative implementation of the baby-step giant-step algorithm for solving the discrete logarithm problem.

---

## Polynomial Arithmetic

A class for performing arithmetic in a quotient polynomial ring $F[x]/\langle \pi(x) \rangle$.

* `class QuotientPolynomialRing(poly, pi_gen)`
    * Represents a polynomial with coefficients as `Fraction` objects, reduced modulo a generator polynomial `pi_gen`.
    * **Methods**:
        * `Add(poly1, poly2)`: Adds two polynomials.
        * `Sub(poly1, poly2)`: Subtracts two polynomials.
        * `Mul(poly1, poly2)`: Multiplies two polynomials.
        * `pow(m)`: Raises the polynomial to the power of `m`.
        * `GCD(poly1, poly2)`: Computes the greatest common divisor of two polynomials.
        * `Inv()`: Computes the multiplicative inverse of a polynomial in the ring.

* `poly_div(num: List[Fraction], den: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]`
    * Performs polynomial long division for polynomials with rational coefficients.

---

## Specialized & Utility Functions

Miscellaneous helper and specialized algorithm functions.

* `is_perfect_power(x: int) -> bool`
    * Checks if an integer $x$ can be expressed as $a^b$ for integers $a > 1, b > 1$.

* `floor_sqrt(x: int) -> int`
    * Computes the integer square root of $x$, i.e., $\lfloor\sqrt{x}\rfloor$.

* `is_smooth(m: int, y: int) -> bool`
    * Determines if an integer $m$ is $y$-smooth, meaning all of its prime factors are less than or equal to $y$.

* `rn_algorithm(n: int) -> list[int]` and `rfn_algorithm(m: int) -> list[tuple[int, int]]`
    * Algorithms for generating a random factored number up to a bound $m$. `rn_algorithm` generates a random non-increasing sequence, which is then used by `rfn_algorithm` to produce a factored number.