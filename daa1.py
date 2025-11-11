def fibonachi_recursive(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = fibonachi_recursive(n - 1)
    fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

n = int(input("Enter a positive integer: "))
result = fibonachi_recursive(n)
print("Fibonacci sequence up to", n, "terms:", result)