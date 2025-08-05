import random

# Define grid dimensions
x_size = 33   # Number of rows
y_size = 32  # Number of columns
num_black_tiles = 24  # Fixed number of black tiles

def create_empty_grid(x_size, y_size):
    """Create an empty grid filled with white tiles ('.')."""
    return [['.' for _ in range(y_size)] for _ in range(x_size)]

def print_grid(grid):
    """Print the grid without any additional characters or highlighting."""
    for row in grid:
        print(''.join(row))
    print()

def place_black_tiles_randomly(grid, num_black_tiles):
    """Randomly place a fixed number of black tiles ('@') on the grid."""
    placed = 0
    while placed < num_black_tiles:
        i = random.randint(0, x_size - 1)
        j = random.randint(0, y_size - 1)
        if grid[i][j] == '.':
            grid[i][j] = '@'
            placed += 1

def place_blue_tiles_adjacent_to_black(grid):
    """
    For each black tile ('@'), ensure at least 2 adjacent blue tiles ('e').
    Tries to place blue tiles in adjacent white cells ('.').
    Returns False if any black tile can't meet the requirement.
    """
    x_size = len(grid)
    y_size = len(grid[0]) if grid else 0
    base_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == '@':
                placed_count = 0

                # Count existing adjacent blue tiles
                for dx, dy in base_directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < x_size and 0 <= nj < y_size and grid[ni][nj] == 'e':
                        placed_count += 1

                # If not enough, try placing new blue tiles
                if placed_count < 2:
                    directions = base_directions.copy()
                    random.shuffle(directions)
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < x_size and 0 <= nj < y_size and grid[ni][nj] == '.':
                            grid[ni][nj] = 'e'
                            placed_count += 1
                            if placed_count >= 2:
                                break

                # If still not enough, fail
                if placed_count < 2:
                    return False

    return True

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
      2. Place fixed black tiles randomly.
      3. Place blue tiles adjacent to the black tiles.
      4. Check connectivity of blue tiles.
    """
    attempt = 0
    import time
    start_time = time.time()
    while True:
        attempt += 1
        print(f"Attempt {attempt}")
        grid = create_empty_grid(x_size, y_size)
        place_black_tiles_randomly(grid, num_black_tiles)
        if not place_blue_tiles_adjacent_to_black(grid):
            #print("Restarting grid generation due to failed blue tile placement.")
            continue  # Restart the whole process if placing blue tiles failed
        add_bottom_row(grid)
        if validate_blue_connectivity(grid):
            # Remove the last row before returning
            grid.pop()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
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
        #print("Re-validating and adjusting layout...")


if __name__ == "__main__":
    # Main program
    grid = correct_layout(num_black_tiles)
    print("Final Grid Layout:")
    print_grid(grid)  # Modify or remove w_rows as needed.
    # Count the number of 'e' symbols
    e_count = sum(row.count('e') for row in grid)
    print(f"Number of 'e' symbols: {e_count}")
    print(f"Final dimensions: {len(grid)} rows x {len(grid[0])} columns")