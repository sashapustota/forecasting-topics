import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

# Set your results directory path where the plots are stored
results_path = 'results/'

# Get a list of all folders (categories) in the results directory
category_folders = [f for f in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, f))]

# Function to create a combined figure for each category
def combine_plots(category_folder):
    # Get the path to the category folder
    category_path = os.path.join(results_path, category_folder)

    # Get all the plot files in this category folder
    plot_files = [f for f in os.listdir(category_path) if f.endswith('.png')]

    # Calculate the grid size for subplots (rows, columns) based on the number of plots
    num_plots = len(plot_files)
    num_cols = 2  # You can change this to adjust the layout
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 6))
    fig.suptitle(f'Linear Regression Fit to Topics in {category_folder}', fontsize=20)

    # Flatten the axes array for easy iteration (in case of multiple rows and columns)
    axs = axs.flatten()

    # Iterate over the plot files and add them to the subplots
    for i, plot_file in enumerate(plot_files):
        # Read the image
        img = imread(os.path.join(category_path, plot_file))

        # Display the image in the subplot
        axs[i].imshow(img)
        axs[i].axis('off')  # Hide the axis

    # Hide any empty subplots if the number of plots is less than the grid size
    for i in range(num_plots, len(axs)):
        axs[i].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the combined figure
    combined_plot_filename = f"{category_folder}_combined.png"
    plt.savefig(os.path.join(category_path, combined_plot_filename), dpi=300)
    plt.close()

    print(f"Saved combined plot for {category_folder}.")

# Iterate over each category folder and create combined plots
for category_folder in category_folders:
    combine_plots(category_folder)

print("All combined plots generated and saved.")
