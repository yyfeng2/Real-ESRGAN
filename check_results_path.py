import os

# Get absolute path of results directory
results_path = os.path.abspath('results')
print(f'Absolute path of results directory: {results_path}')

# Check if directory exists
if os.path.exists(results_path):
    print(f'Results directory exists: {results_path}')
    # List contents
    contents = os.listdir(results_path)
    if len(contents) == 0:
        print('Results directory is empty')
    else:
        print(f'Results directory contents: {contents}')
else:
    print(f'Results directory does not exist: {results_path}')