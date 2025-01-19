import sys
import pandas as pd
import numpy as np

def topsis(data, weights, impacts):
    
    # Step 1: Normalize the Decision Matrix
    norm_matrix = data / np.sqrt((data**2).sum(axis=0))

    # Step 2: Apply Weighting
    weighted_matrix = norm_matrix * weights

    # Step 3: Identify Ideal and Negative-Ideal Solutions
    ideal_solution = np.where(impacts == '+', weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    negative_ideal_solution = np.where(impacts == '+', weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

    # Step 4: Calculate Separation Measures
    dist_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
    dist_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))

    # Step 5: Calculate Relative Closeness to Ideal Solution
    scores = dist_negative_ideal / (dist_ideal + dist_negative_ideal)

    # Step 6: Rank Alternatives
    rankings = scores.argsort()[::-1] + 1  # Higher score -> Higher rank

    # Add results to the original data
    result = data.copy()
    result['Score'] = scores
    result['Rank'] = rankings
    return result

def main():
    # Collect command-line arguments
    input_file = sys.argv[1]
    weights = list(map(float, sys.argv[2].split(',')))  # Convert weights to a list of floats
    impacts = np.array(sys.argv[3].split(','))  # Convert impacts to a list of strings
    result_file = sys.argv[4]

    # Step 1: Load Input Data
    try:
        data = pd.read_excel(input_file, engine='openpyxl')
    except Exception as e:
        print(f"Error: Unable to load data. Details: {e}")
        return

    # Validate data
    if len(data.columns) < 3:
        print("Error: Input file must have at least three columns (alternatives + criteria).")
        return

    if len(weights) != len(data.columns) - 1 or len(impacts) != len(data.columns) - 1:
        print("Error: Number of weights and impacts must match the number of criteria.")
        return

    # Perform TOPSIS
    try:
        decision_matrix = data.iloc[:, 1:]  # Assume first column contains alternatives
        result = topsis(decision_matrix, weights, impacts)
        result.insert(0, data.columns[0], data.iloc[:, 0])  # Add alternatives back to result

        # Save to file
        result.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")
    except Exception as e:
        print(f"Error during TOPSIS analysis: {e}")

if __name__ == "__main__":
    main()
