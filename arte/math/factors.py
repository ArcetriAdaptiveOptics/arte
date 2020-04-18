
def factors(n):
    '''
    Returns a list of factors of *n*.
    No attempt at performance optimisation, use sparingly.
    '''
    factors=[]
    for i in range(2,n//2+1):
        if n % i==0:
           factors.append(i)
    return factors

def gcd(a,b):
    '''
    Returns the greatest common divisor of a and b.
    '''
    while b:
        a, b = b, a % b

    return a

def lcm(a,b):
    '''
    Returns the least common multiple of a and b.
    '''
    return (a*b) // gcd(a,b)
