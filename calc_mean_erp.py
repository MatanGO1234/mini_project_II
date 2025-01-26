import pandas as pd
import matplotlib.pyplot as plt


def calc_mean_erp(trial_points_path, ecog_data_path, pre_event=200, post_event=1000):
    """
    Calculate the mean Event-Related Potential (ERP) for each finger movement.
    Includes correlation calculation and a plot of finger movement vs. brain response.

    Parameters:
    - trial_points_path (str): Path to the CSV file containing trial points.
    - ecog_data_path (str): Path to the CSV file containing ECoG data.
    - pre_event (int): Time in ms to include before the event (default: 200 ms).
    - post_event (int): Time in ms to include after the event (default: 1000 ms).

    Returns:
    - fingers_erp_mean (pd.DataFrame): A 5x1201 DataFrame containing the averaged brain response for each finger.
    """
    # Load the trial points and ECoG data
    trial_points = pd.read_csv(trial_points_path)
    trial_points.columns = ["start", "peak", "finger"]

    # Check for missing values
    if trial_points.isnull().any().any():
        trial_points = trial_points.fillna(trial_points.mean())

    # Ensure all indices are integers
    trial_points["start"] = trial_points["start"].astype(int)
    trial_points["peak"] = trial_points["peak"].astype(int)
    trial_points["finger"] = trial_points["finger"].astype(int)

    ecog_signal = pd.read_csv(ecog_data_path, header=None).values.flatten()

    # Check for outliers in start and peak
    if (trial_points["start"] < 0).any() or (trial_points["start"] >= len(ecog_signal)).any():
        raise ValueError("Some 'start' indices are out of bounds.")
    if (trial_points["peak"] < 0).any() or (trial_points["peak"] >= len(ecog_signal)).any():
        raise ValueError("Some 'peak' indices are out of bounds.")

    # Define the full window size
    window_size = pre_event + post_event + 1

    # Create DataFrame for storing means
    fingers_erp_mean = pd.DataFrame(0.0, index=range(5), columns=range(window_size))  # Initialize with float values
    finger_counts = pd.Series(0.0, index=range(5), dtype=float)  # Initialize with float values

    # Iterate through each trial and process windows
    finger_responses = []
    for _, row in trial_points.iterrows():
        start_idx = row["start"]
        finger_idx = int(row["finger"]) - 1  # Convert to zero-based index

        if start_idx - pre_event >= 0 and start_idx + post_event < len(ecog_signal):
            # Extract the full window surrounding the event
            window = ecog_signal[start_idx - pre_event : start_idx + post_event + 1]

            # Add the window to the respective finger's matrix
            fingers_erp_mean.loc[finger_idx] += pd.Series(window).values
            finger_counts[finger_idx] += 1

            # Save responses for correlation
            finger_responses.append((row["finger"], sum(window)))  # Sum of window as brain response

    # Calculate the mean ERP for each finger
    for i in range(5):
        if finger_counts[i] > 0:
            fingers_erp_mean.loc[i] /= finger_counts[i]

    # Convert finger_responses to a DataFrame
    response_df = pd.DataFrame(finger_responses, columns=["finger", "brain_response"])

    # Calculate correlation
    correlation = response_df["finger"].corr(response_df["brain_response"])
    print(f"Correlation between finger movement and brain response: {correlation}")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(response_df["finger"], response_df["brain_response"], alpha=0.7)
    plt.title("Distribution of Finger Movement vs. Brain Response")
    plt.xlabel("Finger")
    plt.ylabel("Brain Response (Sum of Signal)")
    plt.grid(True)
    plt.show()

    return fingers_erp_mean
