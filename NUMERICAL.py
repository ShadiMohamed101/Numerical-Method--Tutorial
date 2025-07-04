import math
import numpy as np

class NumericalMethods:
    # --- Function Definitions ---
    @staticmethod
    def f(x, function_type=1):
        if function_type == 1:
            return x ** 2 - 3
        elif function_type == 2:
            return math.sin(x)
        elif function_type == 3:
            return math.exp(x) - x - 1
        elif function_type == 4:
            return x ** 3 - 2 * x - 5
        elif function_type == 5:
            return x ** 2
        return x ** 2 - 3  # default case

    # Derivative of the function
    @staticmethod
    def f_prime(x, function_type=1):
        if function_type == 1:
            return 2 * x
        elif function_type == 2:
            return math.cos(x)
        elif function_type == 3:
            return math.exp(x) - 1
        elif function_type == 4:
            return 3 * x ** 2 - 2
        elif function_type == 5:
            return 2 * x
        return 2 * x  # default case

    # Second derivative of the function
    @staticmethod
    def f_double_prime(x, function_type=1):
        if function_type == 1:
            return 2
        elif function_type == 2:
            return -math.sin(x)
        elif function_type == 3:
            return math.exp(x)
        elif function_type == 4:
            return 6 * x
        elif function_type == 5:
            return 2
        return 2  # default case

    # Fixed point function g(x)
    @staticmethod
    def g(x, function_type=1):
        if function_type == 1:
            return math.sqrt(3)  # for f(x) = x^2 - 3
        elif function_type == 2:
            return math.sin(x)
        elif function_type == 3:
            return math.exp(x) - 1
        elif function_type == 4:
            return (2 * x + 5) ** (1 / 3)
        elif function_type == 5:
            return math.sqrt(x)
        return math.sqrt(3)  # default case

    # --- Root Finding Methods ---
    @staticmethod
    def bisection_method(a, b, tol, max_iter, function_type=1):
        product = NumericalMethods.f(a, function_type) * NumericalMethods.f(b, function_type)
        print(f"Check if f(a)*f(b)<0: f({a:.6f})*f({b:.6f}) = {product:.6f}")
        if product >= 0:
            print("Not solvable because f(a)*f(b) >= 0!")
            return None

        print("\nBisection Method")
        print("I \t     a\t\t   b\t\t   c\t\t   f(a)\t\t   f(b)\t\t   f(c)\t\t |f(c)|<ε \t i=N")
        iter_num = 0
        c = a

        while iter_num < max_iter:
            c = (a + b) / 2
            fc = NumericalMethods.f(c, function_type)
            tolerance_achieved = abs(fc) < tol
            reached_max_iterations = (iter_num == max_iter - 1)

            print(f"{iter_num}\t {a:.6f}\t {b:.6f}\t {c:.6f}\t "
                  f"{NumericalMethods.f(a, function_type):.6f}\t "
                  f"{NumericalMethods.f(b, function_type):.6f}\t "
                  f"{fc:.6f}\t {tolerance_achieved}\t {reached_max_iterations}")

            if tolerance_achieved:
                break

            if NumericalMethods.f(c, function_type) * NumericalMethods.f(a, function_type) > 0:
                a = c
            else:
                b = c
                
            iter_num += 1

        print(f"\nx*=c= {c:.6f}")
        return c

    @staticmethod
    def secant_method(x0, x1, tol, max_iter, function_type=1):
        print("\nSecant Method")
        print("I \t     x0\t\t     x1\t\t     x2\t\t   f(x0)\t\t   f(x1)\t\t   f(x2)\t\t |f(x2)|<ε \t i=N")
        iter_num = 0
        fx0 = NumericalMethods.f(x0, function_type)
        fx1 = NumericalMethods.f(x1, function_type)
        x2 = x1

        while iter_num < max_iter:
            if abs(fx0 - fx1) < 1e-10:
                print("Error: Division by near-zero! Method failed.")
                return None

            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            fx2 = NumericalMethods.f(x2, function_type)
            tolerance_achieved = abs(fx2) < tol
            reached_max_iterations = (iter_num == max_iter - 1)

            print(f"{iter_num}\t {x0:.6f}\t {x1:.6f}\t {x2:.6f}\t "
                  f"{fx0:.6f}\t {fx1:.6f}\t {fx2:.6f}\t "
                  f"{tolerance_achieved}\t {reached_max_iterations}")

            if tolerance_achieved:
                break

            x0, x1, fx0, fx1 = x1, x2, fx1, fx2
            iter_num += 1

        print(f"\nx*=c= {x2:.6f}")
        return x2

    @staticmethod
    def modified_secant_method(a, b, tol, max_iter, function_type=1):
        product = NumericalMethods.f(a, function_type) * NumericalMethods.f(b, function_type)
        print(f"Check if f(a)*f(b)<0: f({a:.6f})*f({b:.6f}) = {product:.6f}")
        if product >= 0:
            print("Not solvable because f(a)*f(b) >= 0!")
            return None

        print("\nModified Secant Method")
        print("I \t     a\t\t     b\t\t     c\t\t   f(a)\t\t   f(b)\t\t   f(c)\t\t |f(c)|<ε \t i=N")
        iter_num = 0
        c = b

        while iter_num < max_iter:
            fa = NumericalMethods.f(a, function_type)
            fb = NumericalMethods.f(b, function_type)
            if abs(fb - fa) < 1e-10:
                print("Error: Division by near-zero! Method failed.")
                return None

            c = b - fb * (b - a) / (fb - fa)
            fc = NumericalMethods.f(c, function_type)
            tolerance_achieved = abs(fc) < tol
            reached_max_iterations = (iter_num == max_iter - 1)

            print(f"{iter_num}\t {a:.6f}\t {b:.6f}\t {c:.6f}\t "
                  f"{fa:.6f}\t {fb:.6f}\t {fc:.6f}\t "
                  f"{tolerance_achieved}\t {reached_max_iterations}")

            if tolerance_achieved:
                break

            if fc * fa > 0:
                a = c
            else:
                b = c
                
            iter_num += 1

        print(f"\nx*=c= {c:.6f}")
        return c

    @staticmethod
    def newton_raphson_method(xi, tol, max_iter, function_type=1):
        print("\nNewton-Raphson Method")
        print("I \t     xi\t\t   f(xi)\t\t   f'(xi)\t\t |f(xi)|<ε \t i=N")
        iter_num = 0
        xi_new = xi

        while iter_num < max_iter:
            fxi = NumericalMethods.f(xi, function_type)
            fpxi = NumericalMethods.f_prime(xi, function_type)
            if abs(fpxi) < 1e-10:
                print("Error: Division by near-zero derivative! Method failed.")
                return None

            xi_new = xi - fxi / fpxi
            tolerance_achieved = abs(fxi) < tol
            reached_max_iterations = (iter_num == max_iter - 1)

            print(f"{iter_num}\t {xi:.6f}\t {fxi:.6f}\t {fpxi:.6f}\t "
                  f"{tolerance_achieved}\t {reached_max_iterations}")

            if tolerance_achieved:
                break

            xi = xi_new
            iter_num += 1

        print(f"\nx*=c= {xi_new:.6f}")
        return xi_new

    @staticmethod
    def fixed_point_iteration(xi, tol, max_iter, function_type=1):
        print("\nFixed Point Iteration Method")
        print("I \t     xi\t\t   g(xi)\t\t   f(xi)\t\t |f(xi)|<tol")
        iter_num = 0

        while iter_num < max_iter:
            gxi = NumericalMethods.g(xi, function_type)
            fxi = NumericalMethods.f(xi, function_type)
            tolerance_achieved = abs(fxi) < tol

            print(f"{iter_num}\t{xi:.6f}\t{gxi:.6f}\t{fxi:.6f}\t{tolerance_achieved}")

            if tolerance_achieved:
                print(f"\nConverged to solution after {iter_num} iterations")
                return xi

            xi = gxi
            iter_num += 1

        print(f"\nMaximum iterations reached. Best approximation: {xi:.6f}")
        return xi

    # --- Interpolation Methods ---
    @staticmethod
    def lagrange_interpolation(x_points, y_points, x):
        n = len(x_points)
        result = 0.0
        print("\nLagrange Interpolation")
        print(f"Interpolating at x = {x}")
        print("Basis polynomials:")

        for i in range(n):
            term = y_points[i]
            print(f"\nL_{i}(x) contribution:")
            print(f"Initial term: y_{i} = {y_points[i]}")

            for j in range(n):
                if j != i:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
                    print(f" * (x-x_{j})/(x_{i}-x_{j}) = ({(x - x_points[j]):.4f}/{(x_points[i] - x_points[j]):.4f})")

            print(f"Final term {i}: {term:.6f}")
            result += term

        print(f"\nFinal interpolated value at x = {x}: {result:.6f}")
        return result

    @staticmethod
    def newton_divided_differences(x_points, y_points):
        n = len(x_points)
        coef = [y_points.copy()]

        print("\nNewton Divided Differences Table:")
        print("x\tf(x)\t", end="")
        for i in range(1, n):
            print(f"DD{i}\t", end="")
        print()

        for j in range(1, n):
            coef.append([])
            for i in range(n - j):
                diff = (coef[j - 1][i + 1] - coef[j - 1][i]) / (x_points[i + j] - x_points[i])
                coef[j].append(diff)

        # Print the table
        for i in range(n):
            print(f"{x_points[i]:.4f}\t", end="")
            for j in range(min(i + 1, n)):
                if j < len(coef[j]) and (i - j) < len(coef[j]):
                    print(f"{coef[j][i - j]:.6f}\t", end="")
            print()

        return [coef[i][0] for i in range(n)]

    @staticmethod
    def newton_interpolation(x_points, y_points, x):
        # Calculate divided differences
        coef = NumericalMethods.newton_divided_differences(x_points, y_points)
        n = len(coef)
        result = coef[0]

        print("\nNewton Interpolation")
        print(f"Interpolating at x = {x}")
        print(f"Initial term: {result:.6f}")

        # Calculate and show each term
        terms = [result]
        for i in range(1, n):
            term = coef[i]
            print(f"\nAdding term {i}: {term:.6f}", end="")
            for j in range(i):
                term *= (x - x_points[j])
                print(f" * (x - {x_points[j]:.4f})", end="")
            print(f" = {term:.6f}")
            terms.append(term)
            result += term

        # Show the complete polynomial approximation
        print("\n\nComplete polynomial approximation:")
        print(f"P_{n - 1}({x}) = ", end="")
        for i, term in enumerate(terms):
            if i == 0:
                print(f"{term:.6f}", end="")
            else:
                print(f" + {term:.6f}", end="")
        print(f" = {result:.6f}")

        # Show the actual function value if available (for comparison)
        if len(x_points) == len(y_points) and x in x_points:
            idx = x_points.index(x)
            actual = y_points[idx]
            error = abs(actual - result)
            print(f"\nActual value at x = {x}: {actual:.6f}")
            print(f"Absolute error: {error:.6f}")

        return result

    # --- Integration Methods ---
    @staticmethod
    def basic_midpoint_rule(a, b, function_type=1):
        midpoint = (a + b) / 2
        result = (b - a) * NumericalMethods.f(midpoint, function_type)
        print(f"\nBasic Midpoint Rule")
        print(f"Interval: [{a}, {b}]")
        print(f"Midpoint: {midpoint:.6f}")
        print(f"f(midpoint) = {NumericalMethods.f(midpoint, function_type):.6f}")
        print(f"Result: (b-a)*f(midpoint) = {result:.6f}")
        return result

    @staticmethod
    def basic_trapezoidal_rule(a, b, function_type=1):
        fa = NumericalMethods.f(a, function_type)
        fb = NumericalMethods.f(b, function_type)
        result = (b - a) * (fa + fb) / 2
        print(f"\nBasic Trapezoidal Rule")
        print(f"Interval: [{a}, {b}]")
        print(f"f(a) = {fa:.6f}, f(b) = {fb:.6f}")
        print(f"Result: (b-a)*(f(a)+f(b))/2 = {result:.6f}")
        return result

    @staticmethod
    def basic_simpsons_rule(a, b, function_type=1):
        h = (b - a) / 2
        fa = NumericalMethods.f(a, function_type)
        fb = NumericalMethods.f(b, function_type)
        fm = NumericalMethods.f((a + b) / 2, function_type)
        result = h / 3 * (fa + 4 * fm + fb)
        print(f"\nBasic Simpson's Rule")
        print(f"Interval: [{a}, {b}]")
        print(f"f(a) = {fa:.6f}, f(midpoint) = {fm:.6f}, f(b) = {fb:.6f}")
        print(f"Result: h/3*(f(a) + 4f(midpoint) + f(b)) = {result:.6f}")
        return result

    @staticmethod
    def composite_midpoint_rule(a, b, n, function_type=1):
        h = (b - a) / n
        result = 0
        print(f"\nComposite Midpoint Rule with {n} subintervals")
        print(f"Interval: [{a}, {b}], h = {h:.6f}")

        for i in range(n):
            x_mid = a + (i + 0.5) * h
            fx = NumericalMethods.f(x_mid, function_type)
            result += fx
            print(f"Subinterval {i + 1}: midpoint = {x_mid:.6f}, f(midpoint) = {fx:.6f}")

        result *= h
        print(f"\nFinal result: h * sum = {result:.6f}")
        return result

    @staticmethod
    def composite_trapezoidal_rule(a, b, n, function_type=1):
        h = (b - a) / n
        result = (NumericalMethods.f(a, function_type) + NumericalMethods.f(b, function_type)) / 2
        print(f"\nComposite Trapezoidal Rule with {n} subintervals")
        print(f"Interval: [{a}, {b}], h = {h:.6f}")
        print(f"Initial term: (f(a) + f(b))/2 = {result:.6f}")

        for i in range(1, n):
            x = a + i * h
            fx = NumericalMethods.f(x, function_type)
            result += fx
            print(f"Adding f({x:.6f}) = {fx:.6f}")

        result *= h
        print(f"\nFinal result: h * sum = {result:.6f}")
        return result

    @staticmethod
    def composite_simpsons_rule(a, b, n, function_type=1):
        if n % 2 != 0:
            print("Error: n must be even for Simpson's Rule")
            return None

        h = (b - a) / n
        result = NumericalMethods.f(a, function_type) + NumericalMethods.f(b, function_type)
        print(f"\nComposite Simpson's Rule with {n} subintervals")
        print(f"Interval: [{a}, {b}], h = {h:.6f}")
        print(f"Initial terms: f(a) + f(b) = {result:.6f}")

        for i in range(1, n):
            x = a + i * h
            fx = NumericalMethods.f(x, function_type)
            if i % 2 == 1:
                result += 4 * fx
                print(f"Adding 4*f({x:.6f}) = {4 * fx:.6f}")
            else:
                result += 2 * fx
                print(f"Adding 2*f({x:.6f}) = {2 * fx:.6f}")

        result *= h / 3
        print(f"\nFinal result: h/3 * sum = {result:.6f}")
        return result

    # --- Linear Regression ---
    @staticmethod
    def linear_regression(x_points, y_points):
        n = len(x_points)
        if n != len(y_points):
            print("Error: x and y must have the same length")
            return None

        sum_x = sum(x_points)
        sum_y = sum(y_points)
        sum_xy = sum(x * y for x, y in zip(x_points, y_points))
        sum_x2 = sum(x * x for x in x_points)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            print("Error: Denominator is zero (vertical line)")
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        print("\nLinear Regression Results:")
        print(f"Sum x: {sum_x:.4f}")
        print(f"Sum y: {sum_y:.4f}")
        print(f"Sum xy: {sum_xy:.4f}")
        print(f"Sum x²: {sum_x2:.4f}")
        print(f"Slope (m): {slope:.6f}")
        print(f"Intercept (b): {intercept:.6f}")
        print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")

        return slope, intercept

    # --- Numerical Differentiation ---
    @staticmethod
    def forward_difference(x, h, function_type=1):
        derivative = (NumericalMethods.f(x + h, function_type) - NumericalMethods.f(x, function_type)) / h
        exact = NumericalMethods.f_prime(x, function_type)
        error = abs(exact - derivative)
        
        print("\nForward Difference Method")
        print(f"f'({x}) ≈ [f({x+h}) - f({x})] / {h}")
        print(f"Approximation: {derivative:.8f}")
        print(f"Exact value: {exact:.8f}")
        print(f"Absolute error: {error:.8f}")
        
        return derivative

    @staticmethod
    def backward_difference(x, h, function_type=1):
        derivative = (NumericalMethods.f(x, function_type) - NumericalMethods.f(x - h, function_type)) / h
        exact = NumericalMethods.f_prime(x, function_type)
        error = abs(exact - derivative)
        
        print("\nBackward Difference Method")
        print(f"f'({x}) ≈ [f({x}) - f({x-h})] / {h}")
        print(f"Approximation: {derivative:.8f}")
        print(f"Exact value: {exact:.8f}")
        print(f"Absolute error: {error:.8f}")
        
        return derivative

    @staticmethod
    def central_difference(x, h, function_type=1):
        derivative = (NumericalMethods.f(x + h, function_type) - NumericalMethods.f(x - h, function_type)) / (2 * h)
        exact = NumericalMethods.f_prime(x, function_type)
        error = abs(exact - derivative)
        
        print("\nCentral Difference Method")
        print(f"f'({x}) ≈ [f({x+h}) - f({x-h})] / (2*{h})")
        print(f"Approximation: {derivative:.8f}")
        print(f"Exact value: {exact:.8f}")
        print(f"Absolute error: {error:.8f}")
        
        return derivative

    # --- Direct Methods for Linear Systems ---
    @staticmethod
    def gaussian_elimination(A, b):
        print("\nGaussian Elimination")
        n = len(A)
        
        # Augmented matrix
        M = [A[i] + [b[i]] for i in range(n)]
        
        print("Initial augmented matrix:")
        NumericalMethods.print_matrix(M)
        
        # Forward elimination
        for i in range(n):
            # Partial pivoting
            max_row = i
            for j in range(i+1, n):
                if abs(M[j][i]) > abs(M[max_row][i]):
                    max_row = j
            M[i], M[max_row] = M[max_row], M[i]
            
            # Elimination
            for j in range(i+1, n):
                factor = M[j][i] / M[i][i]
                for k in range(i, n+1):
                    M[j][k] -= factor * M[i][k]
        
        print("\nMatrix after forward elimination:")
        NumericalMethods.print_matrix(M)
        
        # Back substitution
        x = [0] * n
        for i in range(n-1, -1, -1):
            x[i] = M[i][n]
            for j in range(i+1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]
        
        print("\nSolution:")
        for i in range(n):
            print(f"x_{i} = {x[i]:.6f}")
            
        return x

    @staticmethod
    def print_matrix(M):
        for row in M:
            print("[" + "  ".join(f"{elem:8.4f}" for elem in row) + "]")
            
    # --- Iterative Methods for Linear Systems ---
    @staticmethod
    def jacobi_method(A, b, tol=1e-6, max_iter=100):
        print("\nJacobi Iterative Method")
        n = len(A)
        x = [0] * n
        x_new = [0] * n
        
        print("Iteration\t", end="")
        for i in range(n):
            print(f"x_{i}\t\t", end="")
        print("Error")
        
        for k in range(max_iter):
            for i in range(n):
                sigma = 0
                for j in range(n):
                    if j != i:
                        sigma += A[i][j] * x[j]
                x_new[i] = (b[i] - sigma) / A[i][i]
                
            # Calculate error
            error = max(abs(x_new[i] - x[i]) for i in range(n))
            
            # Print current iteration
            print(f"{k}\t\t", end="")
            for val in x_new:
                print(f"{val:.6f}\t", end="")
            print(f"{error:.6f}")
            
            # Check convergence
            if error < tol:
                print(f"\nConverged after {k+1} iterations")
                return x_new
                
            x = x_new.copy()
            
        print(f"\nMaximum iterations reached ({max_iter})")
        return x

    # --- Floating Point Utilities ---
    @staticmethod
    def to_single_precision(x):
        return np.float32(x)

    @staticmethod
    def analyze_floating_point(x):
        x_single = NumericalMethods.to_single_precision(x)
        error = abs(x - x_single)
        
        print(f"Double precision: {x:.15f}")
        print(f"Single precision: {x_single:.15f}")
        print(f"Absolute error: {error:.15f}")
        print(f"Relative error: {error/abs(x):.15f}")
        
        return x_single


# --- Menu System ---
def function_menu():
    print("\nSelect a function:")
    print("1. f(x) = x² - 3")
    print("2. f(x) = sin(x)")
    print("3. f(x) = eˣ - x - 1")
    print("4. f(x) = x³ - 2x - 5")
    print("5. f(x) = x²")
    print("6. Back to main menu")
    return int(input("Enter choice (1-6): "))

def root_finding_menu():
    print("\nRoot Finding Methods:")
    print("1. Bisection Method (Direct)")
    print("2. Secant Method (Indirect)")
    print("3. Modified Secant Method (Indirect)")
    print("4. Newton-Raphson Method (Indirect)")
    print("5. Fixed Point Iteration (Indirect)")
    print("6. Back to main menu")
    return int(input("Enter choice (1-6): "))

def interpolation_menu():
    print("\nInterpolation Methods:")
    print("1. Lagrange Interpolation")
    print("2. Newton Interpolation")
    print("3. Back to main menu")
    return int(input("Enter choice (1-3): "))

def integration_menu():
    print("\nIntegration Methods:")
    print("1. Basic Midpoint Rule")
    print("2. Basic Trapezoidal Rule")
    print("3. Basic Simpson's Rule")
    print("4. Composite Midpoint Rule")
    print("5. Composite Trapezoidal Rule")
    print("6. Composite Simpson's Rule")
    print("7. Back to main menu")
    return int(input("Enter choice (1-7): "))

def differentiation_menu():
    print("\nDifferentiation Methods:")
    print("1. Forward Difference")
    print("2. Backward Difference")
    print("3. Central Difference")
    print("4. Back to main menu")
    return int(input("Enter choice (1-4): "))

def linear_systems_menu():
    print("\nLinear Systems Methods:")
    print("1. Gaussian Elimination (Direct)")
    print("2. Jacobi Method (Iterative)")
    print("3. Back to main menu")
    return int(input("Enter choice (1-3): "))

def floating_point_menu():
    print("\nFloating Point Analysis:")
    print("1. Convert to single precision")
    print("2. Analyze floating point representation")
    print("3. Back to main menu")
    return int(input("Enter choice (1-3): "))

def main():
    while True:
        print("\nMAIN MENU - NUMERICAL METHODS")
        print("1. Root Finding")
        print("2. Interpolation")
        print("3. Linear Regression")
        print("4. Numerical Integration")
        print("5. Numerical Differentiation")
        print("6. Linear Systems Solvers")
        print("7. Floating Point Analysis")
        print("8. Exit")

        choice = int(input("Enter your choice (1-8): "))

        if choice == 1:  # Root Finding
            func_choice = function_menu()
            if func_choice == 6:
                continue

            method_choice = root_finding_menu()
            if method_choice == 6:
                continue

            tol = float(input("Enter tolerance: "))
            max_iter = int(input("Enter maximum iterations: "))

            if method_choice == 1:  # Bisection
                a = float(input("Enter lower bound a: "))
                b = float(input("Enter upper bound b: "))
                NumericalMethods.bisection_method(a, b, tol, max_iter, func_choice)

            elif method_choice == 2:  # Secant
                x0 = float(input("Enter first guess x0: "))
                x1 = float(input("Enter second guess x1: "))
                NumericalMethods.secant_method(x0, x1, tol, max_iter, func_choice)

            elif method_choice == 3:  # Modified Secant
                x0 = float(input("Enter initial guess x0: "))
                delta = float(input("Enter delta (small fraction): "))
                NumericalMethods.modified_secant_method(x0, x0 + delta * x0, tol, max_iter, func_choice)

            elif method_choice == 4:  # Newton-Raphson
                x0 = float(input("Enter initial guess x0: "))
                NumericalMethods.newton_raphson_method(x0, tol, max_iter, func_choice)

            elif method_choice == 5:  # Fixed Point
                x0 = float(input("Enter initial guess x0: "))
                NumericalMethods.fixed_point_iteration(x0, tol, max_iter, func_choice)

        elif choice == 2:  # Interpolation
            method_choice = interpolation_menu()
            if method_choice == 3:
                continue

            x_points = list(map(float, input("Enter x values (space separated): ").split()))
            y_points = list(map(float, input("Enter y values (space separated): ").split()))
            x = float(input("Enter x value to interpolate: "))

            if method_choice == 1:  # Lagrange
                NumericalMethods.lagrange_interpolation(x_points, y_points, x)
            elif method_choice == 2:  # Newton
                NumericalMethods.newton_interpolation(x_points, y_points, x)

        elif choice == 3:  # Linear Regression
            x_points = list(map(float, input("Enter x values (space separated): ").split()))
            y_points = list(map(float, input("Enter y values (space separated): ").split()))
            NumericalMethods.linear_regression(x_points, y_points)

        elif choice == 4:  # Integration
            func_choice = function_menu()
            if func_choice == 6:
                continue

            a = float(input("Enter lower bound a: "))
            b = float(input("Enter upper bound b: "))

            method_choice = integration_menu()
            if method_choice == 7:
                continue

            if method_choice in [1, 2, 3]:  # Basic rules
                if method_choice == 1:
                    NumericalMethods.basic_midpoint_rule(a, b, func_choice)
                elif method_choice == 2:
                    NumericalMethods.basic_trapezoidal_rule(a, b, func_choice)
                elif method_choice == 3:
                    NumericalMethods.basic_simpsons_rule(a, b, func_choice)

            elif method_choice in [4, 5, 6]:  # Composite rules
                n = int(input("Enter number of subintervals (n): "))
                if method_choice == 4:
                    NumericalMethods.composite_midpoint_rule(a, b, n, func_choice)
                elif method_choice == 5:
                    NumericalMethods.composite_trapezoidal_rule(a, b, n, func_choice)
                elif method_choice == 6:
                    NumericalMethods.composite_simpsons_rule(a, b, n, func_choice)

        elif choice == 5:  # Differentiation
            func_choice = function_menu()
            if func_choice == 6:
                continue

            x = float(input("Enter the point x: "))
            h = float(input("Enter step size h: "))

            method_choice = differentiation_menu()
            if method_choice == 4:
                continue

            if method_choice == 1:
                NumericalMethods.forward_difference(x, h, func_choice)
            elif method_choice == 2:
                NumericalMethods.backward_difference(x, h, func_choice)
            elif method_choice == 3:
                NumericalMethods.central_difference(x, h, func_choice)

        elif choice == 6:  # Linear Systems
            method_choice = linear_systems_menu()
            if method_choice == 3:
                continue
                
            n = int(input("Enter number of equations: "))
            print("Enter coefficient matrix A (row by row):")
            A = []
            for i in range(n):
                row = list(map(float, input(f"Row {i+1} (space separated): ").split()))
                A.append(row)
                
            b = list(map(float, input("Enter right-hand side vector b (space separated): ").split()))
            
            if method_choice == 1:
                NumericalMethods.gaussian_elimination(A, b)
            elif method_choice == 2:
                tol = float(input("Enter tolerance: "))
                max_iter = int(input("Enter maximum iterations: "))
                NumericalMethods.jacobi_method(A, b, tol, max_iter)
                
        elif choice == 7:  # Floating Point
            method_choice = floating_point_menu()
            if method_choice == 3:
                continue
                
            x = float(input("Enter a floating point number: "))
            
            if method_choice == 1:
                result = NumericalMethods.to_single_precision(x)
                print(f"Single precision value: {result}")
            elif method_choice == 2:
                NumericalMethods.analyze_floating_point(x)

        elif choice == 8:  # Exit
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
