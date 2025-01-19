import sys
import pandas as pd
import numpy as np

def perform_topsis(matrix, criteria_weights, criteria_impacts):
    # Step 1: Normalize the Decision Matrix
    normalized_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))

    # Step 2: Apply Weights
    weighted_matrix = normalized_matrix * criteria_weights

    # Step 3: Determine Ideal and Negative-Ideal Solutions
    best_solution = np.where(criteria_impacts == '+', weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    worst_solution = np.where(criteria_impacts == '+', weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

    # Step 4: Compute Separation Distances
    distance_to_best = np.sqrt(((weighted_matrix - best_solution) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((weighted_matrix - worst_solution) ** 2).sum(axis=1))

    # Step 5: Calculate Closeness to Ideal Solution
    relative_closeness = distance_to_worst / (distance_to_best + distance_to_worst)

    # Step 6: Rank the Alternatives
    rankings = relative_closeness.argsort()[::-1] + 1  # Higher closeness -> Higher rank

    # Append scores and rankings to the original matrix
    results = matrix.copy()
    results['Closeness'] = relative_closeness
    results['Ranking'] = rankings
    return results

def main():
    # Read arguments from the command line
    file_path = sys.argv[1]
    criteria_weights = list(map(float, sys.argv[2].split(',')))  # Convert weights to a float list
    criteria_impacts = np.array(sys.argv[3].split(','))  # Split impacts into a string array
    output_file = sys.argv[4]

    # Step 1: Load the input data
    try:
        decision_data = pd.read_excel(file_path, engine='openpyxl')
    except Exception as error:
        print(f"Error: Unable to read the input file. Details: {error}")
        return

    # Validate the input data
    if len(decision_data.columns) < 3:
        print("Error: The input file should have at least three columns (alternatives + criteria).")
        return

    if len(criteria_weights) != len(decision_data.columns) - 1 or len(criteria_impacts) != len(decision_data.columns) - 1:
        print("Error: The number of weights and impacts must match the number of criteria.")
        return

    # Perform TOPSIS analysis
    try:
        criteria_matrix = decision_data.iloc[:, 1:]  # Exclude the first column (assume alternatives)
        analysis_result = perform_topsis(criteria_matrix, criteria_weights, criteria_impacts)
        analysis_result.insert(0, decision_data.columns[0], decision_data.iloc[:, 0])  # Reattach alternatives

        # Save the results to a file
        analysis_result.to_csv(output_file, index=False)
        print(f"The TOPSIS analysis results have been successfully saved to {output_file}.")
    except Exception as error:
        print(f"Error occurred during the TOPSIS analysis. Details: {error}")

if __name__ == "__main__":
    main()

