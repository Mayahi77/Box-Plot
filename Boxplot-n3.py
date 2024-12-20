import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def calculate_statistics(data):
    return {
        "Mean": np.mean(data),
        "Std Dev": np.std(data),
        "Median": np.median(data),
        "Min": np.min(data),
        "Max": np.max(data)
    }

def process_files_grouped(initial_files, four_hour_files, forty_eight_hour_files, speeds, sample_size):
    grouped_data = []
    statistics = []

    for i, speed in enumerate(speeds):
        data_for_speed = []
        stats_for_speed = []

        # Process initial, 4h, and 48h for each speed
        for files, label in zip([initial_files, four_hour_files, forty_eight_hour_files], ["Initial", "4h", "48h"]):
            file = files[i]
            try:
                df = pd.read_csv(file, delimiter="\t")
                df = df.drop(columns=["Unnamed: 3"], errors="ignore")  # Drop unnecessary columns

                if "Actual Torque [of nominal]" not in df.columns:
                    st.warning(f"The file {file.name} does not contain the column `Actual Torque [of nominal]`.")
                    return None, None

                torque_data = df["Actual Torque [of nominal]"].dropna()

                # Ensure consistent sample size
                if len(torque_data) >= sample_size:
                    sampled_data = torque_data.sample(n=sample_size, random_state=42)
                else:
                    st.warning(f"File {file.name} has fewer than {sample_size} data points. Taking all available data.")
                    sampled_data = torque_data

                # Remove outliers using IQR
                torque_data_cleaned = remove_outliers_iqr(sampled_data)

                # Append data and statistics
                data_for_speed.append(torque_data_cleaned)
                stats_for_speed.append(calculate_statistics(torque_data_cleaned))
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")
                return None, None

        grouped_data.append(data_for_speed)
        statistics.append(stats_for_speed)

    return grouped_data, statistics

def plot_grouped_boxplot_with_stats(grouped_data, speeds, test_labels, statistics):
    fig, ax = plt.subplots(figsize=(10, 6))
    num_tests = len(test_labels)
    positions = []
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Calculate positions for grouped box plots
    for i, speed in enumerate(speeds):
        base_pos = i * (num_tests + 1)
        positions.extend([base_pos + j for j in range(num_tests)])

    # Flatten data for plotting
    all_data = [data for group in grouped_data for data in group]

    # Create box plot
    boxplots = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        whis=(0, 100)
    )

    # Apply colors
    for patch, position in zip(boxplots['boxes'], positions):
        patch.set_facecolor(colors[position % num_tests])

    # Set x-axis labels
    group_centers = [i * (num_tests + 1) + (num_tests - 1) / 2 for i in range(len(speeds))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(s) for s in speeds])
    ax.set_xlabel("Speed [rad/s]")
    ax.set_ylabel("Actual Torque [of nominal]")
    ax.set_title("Grouped Box Plot")

    # Set y-axis range
    ax.set_ylim(-1.25, 0.75)
    ax.set_yticks(np.arange(-1.25, 0.76, 0.25))

    # Add legend
    for i, label in enumerate(test_labels):
        ax.plot([], [], label=label, linestyle='none', marker='s', markersize=10, color=colors[i])
    ax.legend(title="Tests", loc="upper right")

    # Add statistics table
    stats_table_data = []
    columns = ["Speed", "Test", "Mean", "Std Dev", "Median", "Min", "Max"]
    for i, speed in enumerate(speeds):
        for j, label in enumerate(test_labels):
            stats = statistics[i][j]
            stats_table_data.append([
                speed,
                label,
                stats["Mean"],
                stats["Std Dev"],
                stats["Median"],
                stats["Min"],
                stats["Max"]
            ])

    stats_df = pd.DataFrame(stats_table_data, columns=columns)
    st.table(stats_df)

    return fig

def main():
    st.title("Consistent Grouped Box Plot with Sample Size and Summary Table")
    
    speeds = [0.25, 0.5, 1.0, 1.5, 2.0]
    test_labels = ["Initial", "4h", "48h"]

    # Input for sample size
    sample_size = st.number_input(
        "Enter the sample size (n) for consistent analysis:",
        min_value=1,
        value=10,
        step=1,
        help="Specify the number of random samples to take from each file."
    )

    # Upload files for each test
    st.subheader("Upload Files for Initial Test")
    initial_files = st.file_uploader(
        "Upload files in order (0.25, 0.5, 1.0, 1.5, 2.0)", accept_multiple_files=True, type=["csv"], key="initial"
    )
    st.subheader("Upload Files for 4h Test")
    four_hour_files = st.file_uploader(
        "Upload files in order (0.25, 0.5, 1.0, 1.5, 2.0)", accept_multiple_files=True, type=["csv"], key="4h"
    )
    st.subheader("Upload Files for 48h Test")
    forty_eight_hour_files = st.file_uploader(
        "Upload files in order (0.25, 0.5, 1.0, 1.5, 2.0)", accept_multiple_files=True, type=["csv"], key="48h"
    )

    if st.button("Generate Plot and Table"):
        if initial_files and four_hour_files and forty_eight_hour_files:
            if len(initial_files) == len(four_hour_files) == len(forty_eight_hour_files) == len(speeds):
                grouped_data, statistics = process_files_grouped(initial_files, four_hour_files, forty_eight_hour_files, speeds, sample_size)
                if grouped_data:
                    fig = plot_grouped_boxplot_with_stats(grouped_data, speeds, test_labels, statistics)
                    st.pyplot(fig)
            else:
                st.warning("Please upload the correct number of files for each test (5 files).")
        else:
            st.warning("Please upload files for all three tests before generating the plot and table.")

if __name__ == "__main__":
    main()
