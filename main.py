from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2  # Correct import path

def angle(bias):
    # For a single qubit, the angle of rotation (theta) is related to the probability of measuring 1.
    # cos(theta/2)^2 = bias (probability of 1), so theta = 2 * acos(sqrt(bias))
    return 2 * np.arccos(np.sqrt(bias))

# Function to get valid bias input from user
def get_valid_bias():
    while True:
        try:
            bias = float(input("What is the probability of getting 0? (between 0 and 1): "))
            if 0 <= bias <= 1:
                return bias
            else:
                print("Invalid input! Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

# Initialize quantum and classical registers
qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

# Get valid user input for the bias
bias = get_valid_bias()

# Apply Ry rotation gate with a custom angle based on bias
circuit.ry(angle(bias), qreg_q[0])

# Measure the qubit
circuit.measure(qreg_q[0], creg_c[0])

# Backend setup for simulation (Fake Backend)
backend = FakeAlmadenV2()

# Execute the circuit on the backend
job = backend.run(circuit)

# Get the result from the job
result = job.result()

# Get the counts from the result
result_counts = result.get_counts(circuit)
observed_1 = result_counts.get('1', 0)
observed_0 = result_counts.get('0', 0)
total = observed_1 + observed_0

# Theoretical probabilities of 1 and 0
expected_0 = bias*total # Probability of 0 based on the input bias
expected_1 = (1-bias)*total  # Probability of 1

# Handle division by zero for percent difference calculations
def safe_percent_diff(expected, observed):
    if expected == 0:
        return "∞"  # Return infinity symbol if expected is zero
    else:
        return 100 * abs(expected - observed) / expected

# Print the results and the probabilistic analysis
print(f"Observed counts: {observed_0} 0's, and {observed_1} 1's")
print(f"Expected counts: {expected_0} 0's, and {expected_1} 1's")
print(f"Percent error: {safe_percent_diff(expected_0, observed_0)}% for 0, and {safe_percent_diff(expected_1, observed_1)}% for 1")

# Plot the results and analysis
# Set the width of the bars
bar_width = 0.35

# Set the x positions for the bars
x = np.arange(2)

# Plot observed bars, shifting them by bar_width
plt.bar(x - bar_width / 2, [observed_0, observed_1], bar_width, alpha=0.6, label="Observed")

# Plot expected bars, shifting them by -bar_width to place them next to the observed bars
plt.bar(x + bar_width / 2, [expected_0, expected_1], bar_width, alpha=0.3, label="Expected", color="red")

# Customize the plot
plt.xlabel("Measured Value")
plt.ylabel("Probability")
plt.title("Observed vs Expected Probabilities")
plt.xticks(x, ['0', '1'])  # Set x-ticks to '0' and '1'
plt.legend()

# Show the plot
plt.show()