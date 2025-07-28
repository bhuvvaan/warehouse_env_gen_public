import random
import json
import os
import time
from tqdm import tqdm

# Define grid dimensions
x_size = 33
y_size = 32
num_black_tiles = 240

def create_empty_grid(x_size, y_size):
    return [['.' for _ in range(y_size)] for _ in range(x_size)]

def print_grid(grid):
    for row in grid:
        print(''.join(row))
    print()

def print_grid_with_highlight(grid, highlight_positions=None, highlight_char='X'):
    """Print grid with highlighted problematic positions"""
    if highlight_positions is None:
        highlight_positions = set()
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if (i, j) in highlight_positions:
                print(highlight_char, end='')
            else:
                print(cell, end='')
        print()
    print()

def place_black_tiles_randomly(grid, num_black_tiles):
    placed = 0
    while placed < num_black_tiles:
        i = random.randint(0, x_size - 1)
        j = random.randint(0, y_size - 1)
        if grid[i][j] == '.':
            grid[i][j] = '@'
            placed += 1

def place_blue_tiles_adjacent_to_black(grid):
    x_size = len(grid)
    y_size = len(grid[0]) if grid else 0
    base_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    problematic_position = None

    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == '@':
                placed_count = 0
                for dx, dy in base_directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < x_size and 0 <= nj < y_size and grid[ni][nj] == 'e':
                        placed_count += 1
                
                if placed_count < 2:
                    directions = base_directions.copy()
                    random.shuffle(directions)
                    available_spots = []
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < x_size and 0 <= nj < y_size and grid[ni][nj] == '.':
                            available_spots.append((ni, nj))
                    
                    # If we can't place enough blue tiles
                    if len(available_spots) + placed_count < 2:
                        print(f"\nProblem at black tile ({i}, {j}):")
                        print(f"Only {placed_count} blue tiles placed and {len(available_spots)} possible spots available")
                        print("Current grid state around this black tile:")
                        # Print a small region around the problematic black tile
                        highlight_positions = {(i, j)}  # The black tile
                        for dx, dy in base_directions:
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < x_size and 0 <= nj < y_size:
                                highlight_positions.add((ni, nj))
                        
                        # Print a 5x5 region centered on the problematic tile
                        region_size = 2
                        min_i = max(0, i - region_size)
                        max_i = min(x_size, i + region_size + 1)
                        min_j = max(0, j - region_size)
                        max_j = min(y_size, j + region_size + 1)
                        
                        subgrid = [row[min_j:max_j] for row in grid[min_i:max_i]]
                        print_grid_with_highlight(subgrid, {(i-min_i, j-min_j)}, 'P')
                        return False, (i, j)  # Return the problematic position
                    
                    # Place the blue tiles
                    for ni, nj in available_spots[:2-placed_count]:
                        grid[ni][nj] = 'e'
                        placed_count += 1
                        if placed_count >= 2:
                            break
    return True, None

def refresh_black_tiles_in_region(grid, center_i, center_j, region_size=2):
    """Refresh black tile positions only in a specific region around the center point"""
    # Calculate region boundaries
    min_i = max(0, center_i - region_size)
    max_i = min(len(grid), center_i + region_size + 1)
    min_j = max(0, center_j - region_size)
    max_j = min(len(grid[0]), center_j + region_size + 1)
    
    # Count black tiles in the region
    black_count = 0
    for i in range(min_i, max_i):
        for j in range(min_j, max_j):
            if grid[i][j] == '@':
                black_count += 1
                grid[i][j] = '.'  # Remove the black tile
    
    # Place the same number of black tiles randomly in the region
    placed = 0

    while placed < black_count:
        i = random.randint(min_i, max_i - 1)
        j = random.randint(min_j, max_j - 1)
        if grid[i][j] == '.':
            grid[i][j] = '@'
            placed += 1

            
    return placed == black_count  # Return True if all black tiles were successfully placed

def validate_blue_connectivity(grid):
    visited = [[False for _ in range(y_size)] for _ in range(x_size)]
    disconnected_blues = []

    def dfs(i, j):
        if i < 0 or i >= x_size or j < 0 or j >= y_size:
            return
        if grid[i][j] == '@' or visited[i][j]:
            return
        visited[i][j] = True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(i + dx, j + dy)

    start_found = False
    start_pos = None
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == 'e':
                dfs(i, j)
                start_found = True
                start_pos = (i, j)
                break
        if start_found:
            break

    if not start_found:
        print("\nNo blue tiles found in the grid!")
        return False

    # Find disconnected blue tiles
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == 'e' and not visited[i][j]:
                disconnected_blues.append((i, j))

    if disconnected_blues:
        print("\nFound disconnected blue tiles!")
        print(f"Starting blue tile was at: {start_pos}")
        print("Disconnected blue tiles at:", disconnected_blues)
        
        # Print the region around the first disconnected tile
        i, j = disconnected_blues[0]
        region_size = 3
        min_i = max(0, i - region_size)
        max_i = min(x_size, i + region_size + 1)
        min_j = max(0, j - region_size)
        max_j = min(y_size, j + region_size + 1)
        
        print("\nRegion around first disconnected tile:")
        subgrid = [row[min_j:max_j] for row in grid[min_i:max_i]]
        highlight_positions = {(i-min_i, j-min_j)}
        print_grid_with_highlight(subgrid, highlight_positions, 'D')
        return False

    return True

def count_black_tiles(grid):
    """Count the number of black tiles in the grid"""
    return sum(row.count('@') for row in grid)

def remove_blue_tiles(grid):
    """Remove all blue tiles from the grid, replacing them with empty spaces"""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 'e':
                grid[i][j] = '.'

def promote_problematic_black(grid, problem_pos):
    """Convert the black tile at problem_pos to a blue tile and turn a random '.' tile into a black tile to keep counts balanced."""
    i, j = problem_pos
    if grid[i][j] != '@':
        # The tile is no longer black; nothing to do
        return False

    # Promote the problematic black tile to blue
    grid[i][j] = 'e'

    # Find a random '.' tile to turn into a black tile
    max_attempts = 1000
    attempts = 0
    while attempts < max_attempts:
        ri = random.randint(0, len(grid) - 1)
        rj = random.randint(0, len(grid[0]) - 1)
        if grid[ri][rj] == '.':
            grid[ri][rj] = '@'
            return True
        attempts += 1
    # If we couldn't find any empty spot (very unlikely), revert the change
    grid[i][j] = '@'
    return False

def retry_problematic_grid(grid, grid_id=None, problem_pos=None):
    """Attempt to fix a problematic grid by converting the problematic black tile to blue
    and turning a random empty tile into a black tile. This keeps the black-tile count
    constant while removing the immediate problem."""
    if problem_pos is None:
        print("\nNo problem position provided, cannot perform local promotion fix")
        return grid, False

    max_retries = 50
    for attempt in range(max_retries):
        print(f"\nAttempting promotion fix around position {problem_pos} – attempt {attempt + 1}/{max_retries}")

        if not promote_problematic_black(grid, problem_pos):
            print("Promotion failed (tile no longer black or no empty spot found), aborting this attempt")
            return grid, False

        # After promotion, try to satisfy adjacency for all black tiles
        success, new_problem_pos = place_blue_tiles_adjacent_to_black(grid)
        if not success:
            print("Still found problematic black tile, will promote that one next")
            problem_pos = new_problem_pos
            continue  # Try again within retry loop

        # All black tiles have enough adjacent blues; now check connectivity
        print("Checking blue tile connectivity…")
        if validate_blue_connectivity(grid):
            print("Success! Grid is now valid after promotion fix")
            return grid, True
        else:
            print("Connectivity failed after promotion fix – abandoning this grid.")
            return grid, False  # Give up on this grid; outer loop will start a fresh one

def correct_layout(num_black_tiles, grid_id=None):
    attempt = 0
    while True:
        attempt += 1
        tqdm.write(f"Grid {grid_id} - Attempt {attempt}")
        grid = create_empty_grid(x_size, y_size)
        place_black_tiles_randomly(grid, num_black_tiles)
        
        success, problem_pos = place_blue_tiles_adjacent_to_black(grid)
        if not success:
            print(f"Failed to place blue tiles, will retry")
            # Try to fix the grid
            fixed_grid, success = retry_problematic_grid(grid, grid_id, problem_pos)
            if success:
                grid = fixed_grid
            else:
                continue
        
        # Check connectivity
        if not validate_blue_connectivity(grid):
            print("Failed to validate blue connectivity, starting a fresh grid…")
            continue  # Abandon this grid and start over
        
        # Add border and walls
        for row in grid:
            row.insert(0, '.')
            row.append('.')
        for i, row in enumerate(grid):
            row.append('w' if i % 3 == 1 else '.')
        for i, row in enumerate(grid):
            row.insert(0, 'w' if i % 3 == 1 else '.')
        return grid

def generate_grid(grid_id, verbose=False):
    start = time.time()
    grid = correct_layout(num_black_tiles, grid_id=grid_id)
    e_count = sum(row.count('e') for row in grid)
    end = time.time()

    if verbose:
        tqdm.write(
            f"({len(grid)}x{len(grid[0])})"
        )
        tqdm.write(f"\nGrid {grid_id} layout:")
        print_grid(grid)

    # Flatten the grid into a single string for a more compact JSON representation
    flat_grid_str = ''.join(cell for row in grid for cell in row)

    return {
        "grid_id": grid_id,
        "grid": flat_grid_str,
        "e_count": e_count
    }

if __name__ == "__main__":
    # Ensure output directory (map_generation) exists – it does because this script is inside it.
    output_file = os.path.join(os.path.dirname(__file__), "warehouse_grids_2.json")

    NUM_GRIDS = 10000
    VERBOSE = False  # Printing every grid for 10k envs is overwhelming; switch off by default

    start_time = time.time()

    with open(output_file, 'w') as f:
        f.write('{"grids":[\n')

        for grid_id in tqdm(range(8833, NUM_GRIDS + 1), desc="Generating grids"):
            try:
                result = generate_grid(grid_id, VERBOSE)
                # Write comma separator after the first record
                if grid_id > 8833:
                    f.write(',\n')
                json.dump(result, f)
                f.flush()  # Ensure data is written to disk incrementally
            except Exception as e:
                tqdm.write(f"❌ Error generating Grid {grid_id}: {e}")

        f.write(']}')  # Close the JSON array and object

    total_time = time.time() - start_time
    print(f"\n✅ Completed {NUM_GRIDS} grids in {total_time:.2f} seconds.")
    print(f"📝 Saved to: {output_file}")
