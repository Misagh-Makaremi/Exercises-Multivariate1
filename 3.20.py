import numpy as np

# Define the matrix A
A = np.array([[2, 1], [1, 3]])

# Calculate eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)

# Construct diagonal matrix D with square root of eigenvalues
D_sqrt = np.diag(np.sqrt(eigenvalues))

# Inverse of eigenvectors matrix
P_inv = np.linalg.inv(eigenvectors)

# Calculate A^(1/2) using the formula: A^(1/2) = P * D^(1/2) * P^(-1)
A_sqrt = eigenvectors.dot(D_sqrt).dot(P_inv)

print("Square root of matrix A:")
print(A_sqrt)

# Construct diagonal matrix D_inv_sqrt with inverse square root of eigenvalues
D_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues))

# Inverse of eigenvectors matrix
P_inv = np.linalg.inv(eigenvectors)

# Calculate A^(-1/2) using the formula: A^(-1/2) = P * D^(-1/2) * P^(-1)
A_inv_sqrt = eigenvectors.dot(D_inv_sqrt).dot(P_inv)

print("Inverse square root of matrix A:")
print(A_inv_sqrt)

# Calculate the product A^(1/2) * A^(-1/2)
product_1 = np.dot(A_sqrt, A_inv_sqrt)

# Calculate the product A^(-1/2) * A^(1/2)
product_2 = np.dot(A_inv_sqrt, A_sqrt)

# Identity matrix
identity_matrix = np.eye(2)

# Check if product_1 and product_2 are equal to the identity matrix
print("A^(1/2) * A^(-1/2):")
print(product_1)
print("\nA^(-1/2) * A^(1/2):")
print(product_2)
print("\nAre they equal to the identity matrix?")
print(np.allclose(product_1, identity_matrix) and np.allclose(product_2, identity_matrix))