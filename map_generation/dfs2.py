import random

# Define grid dimensions
x_size = 33   # Number of rows
y_size = 32  # Number of columns
num_black_tiles = 240  # Fixed number of black tiles
seed = 2815

# Create two RNGs: one fixed for black, one random for blue
def get_rngs():
    black_rng = random.Random(seed)  # Fixed seed for black tiles
    blue_rng = random.Random()        # Random seed for blue tiles (system time)
    return black_rng, blue_rng

def create_empty_grid(x_size, y_size):
    """Create an empty grid filled with white tiles ('.')."""
    return [['.' for _ in range(y_size)] for _ in range(x_size)]

def print_grid(grid):
    """Print the grid without any additional characters or highlighting."""
    for row in grid:
        print(''.join(row))
    print()

def place_black_tiles_randomly(grid, num_black_tiles, rng):
    """Randomly place a fixed number of black tiles ('@') on the grid using the provided rng."""
    placed = 0
    while placed < num_black_tiles:
        i = rng.randint(0, x_size - 1)
        j = rng.randint(0, y_size - 1)
        if grid[i][j] == '.':
            grid[i][j] = '@'
            placed += 1

def place_blue_tiles_adjacent_to_black(grid, rng):
    """
    For each black tile, attempt to place a blue tile ('e') in one of its adjacent cells.
    Each black tile gets one blue tile (if an adjacent white cell is available).
    If any black tile cannot have a blue tile placed next to it, restart the whole process.
    Uses the provided rng for shuffling directions.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == '@':
                rng.shuffle(directions)
                placed = False
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < x_size and 0 <= nj < y_size and grid[ni][nj] == '.':
                        grid[ni][nj] = 'e'
                        placed = True
                        break  # Place only one blue tile per black tile.
                if not placed:
                    return False  # Indicate failure and restart the process
    return True  # Return True if all black tiles have adjacent blue tiles

def add_bottom_row(grid):
    """Adds a row at the bottom with the first tile as 'e' and the rest as '.'. 
    This is to ensure that the blue tiles are connected to Goal"""
    new_row = ['e'] + ['.'] * (y_size - 1)  # First tile is 'e', others are '.'
    grid.append(new_row)  # Add new row to the grid


def validate_blue_connectivity(grid):
    """
    Check that all blue tiles ('e') are connected, treating black tiles ('@') as barriers.
    We use a DFS starting from the first blue tile found.
    """
    visited = [[False for _ in range(y_size)] for _ in range(x_size)]
    
    def dfs(i, j):
        if i < 0 or i >= x_size or j < 0 or j >= y_size:
            return
        if grid[i][j] == '@' or visited[i][j]:
            return
        visited[i][j] = True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(i + dx, j + dy)
    
    # Find the first blue tile to start DFS.
    start_found = False
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == 'e':
                dfs(i, j)
                start_found = True
                break
        if start_found:
            break

    # If no blue tile is found, connectivity fails.
    if not start_found:
        return False

    # Ensure every blue tile was visited by DFS.
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == 'e' and not visited[i][j]:
                return False
    return True


def correct_layout(num_black_tiles):
    """
    Repeatedly generate the grid until the blue tiles ('e') are fully connected.
    The process:
      1. Create an empty grid.
      2. Place fixed black tiles randomly (with fixed seed).
      3. Place blue tiles adjacent to the black tiles (randomized each run).
      4. Check connectivity of blue tiles.
    """
    black_rng, blue_rng = get_rngs()
    while True:
        print("Re-initializing RNGs...")
        black_rng = random.Random(seed)  # Re-initialize every time!

        grid = create_empty_grid(x_size, y_size)
        place_black_tiles_randomly(grid, num_black_tiles, black_rng)
        print("Placed black tiles")
        if not place_blue_tiles_adjacent_to_black(grid, blue_rng):
            print("Failed to place blue tiles")
            continue
        add_bottom_row(grid)
        print("Added bottom row")
        if validate_blue_connectivity(grid):
            # Remove the last row before returning
            grid.pop()
            
            # Add empty column on both sides
            for row in grid:
                row.insert(0, '.')  # Add empty column on the left
                row.append('.')     # Add empty column on the right
            # Add another column with 'w' at multiples of 3
            for i, row in enumerate(grid):
                if i % 3 == 1:
                    row.append('w')
                else:
                    row.append('.')
            for i, row in enumerate(grid):
                if i % 3 == 1:
                    row.insert(0, 'w')
                else:
                    row.insert(0, '.')     
            return grid


if __name__ == "__main__":
    # Main program
    grid = correct_layout(num_black_tiles)
    print("Final Grid Layout:")
    print_grid(grid)  # Modify or remove w_rows as needed.
    # Count the number of 'e' symbols
    e_count = sum(row.count('e') for row in grid)
    print(f"Number of 'e' symbols: {e_count}")
    print(f"Final dimensions: {len(grid)} rows x {len(grid[0])} columns")