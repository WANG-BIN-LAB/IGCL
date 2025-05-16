import pandas as pd
import re

# Configuration parameters
DATA_FILE = '.txt'
OUTPUT_FILE = '.xlsx'
MATRIX_SIZE = 9


def extract_accuracies(file_path):
    """Extract accuracy values from text file using regex pattern"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [float(re.search(r'(\d\.\d+)', line).group(1)) for line in lines]


def build_symmetric_matrix(accuracies, size=MATRIX_SIZE):
    """Construct symmetric matrix from accuracy data"""
    matrix = [[None] * size for _ in range(size)]
    index = 0
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i][j] = accuracies[index]
            matrix[j][i] = accuracies[index + 1]
            index += 2
    return matrix


def main():
    # Data processing pipeline
    accuracies = extract_accuracies(DATA_FILE)
    correlation_matrix = build_symmetric_matrix(accuracies)
    df = pd.DataFrame(correlation_matrix)

    # Save results
    df.to_excel(OUTPUT_FILE, index=False, header=False)
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
