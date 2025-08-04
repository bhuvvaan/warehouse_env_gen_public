#!/usr/bin/env python3

def string_to_grid(layout_string):
    """Convert a string layout to a 2D grid.
    The string should be a single line with each row separated by newlines."""
    return [list(row) for row in layout_string.strip().split('\n')]

def validate_blue_connectivity(grid):
    """
    Check that all blue tiles ('e') are connected, treating black tiles ('@') as barriers.
    Workstation tiles ('w') and empty spaces ('.') can be traversed.
    Uses DFS starting from the first blue tile found.
    
    Args:
        grid: 2D list representing the layout
    
    Returns:
        dict: Information about the validation and tile statistics
    """
    x_size = len(grid)
    y_size = len(grid[0]) if grid else 0
    
    # Create visited array
    visited = [[False for _ in range(y_size)] for _ in range(x_size)]
    
    # Count all tile types
    tile_counts = {
        'e': 0,  # blue tiles (endpoints)
        '@': 0,  # black tiles (obstacles)
        'w': 0,  # workstation tiles
        '.': 0   # empty spaces
    }
    
    for row in grid:
        for cell in row:
            if cell in tile_counts:
                tile_counts[cell] += 1
    
    visited_blue_tiles = 0
    
    def dfs(i, j):
        nonlocal visited_blue_tiles
        if i < 0 or i >= x_size or j < 0 or j >= y_size:
            return
        if grid[i][j] == '@' or visited[i][j]:
            return
        
        visited[i][j] = True
        if grid[i][j] == 'e':
            visited_blue_tiles += 1
            
        # Check all four directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_i, new_j = i + dx, j + dy
            dfs(new_i, new_j)
    
    # Find the first blue tile to start DFS
    start_found = False
    start_pos = None
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == 'e':
                start_pos = (i, j)
                dfs(i, j)
                start_found = True
                break
        if start_found:
            break

    # Find any unvisited blue tiles
    unvisited_blue = []
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == 'e' and not visited[i][j]:
                unvisited_blue.append((i, j))
    
    # Prepare result information
    result = {
        'is_connected': visited_blue_tiles == tile_counts['e'],
        'tile_counts': tile_counts,
        'visited_blue_tiles': visited_blue_tiles,
        'start_position': start_pos,
        'dimensions': (x_size, y_size),
        'unvisited_blue_tiles': unvisited_blue
    }
    
    return result

def print_grid_stats(grid):
    """Print detailed statistics about the grid layout"""
    x_size = len(grid)
    y_size = len(grid[0]) if grid else 0
    
    print("\nGrid Statistics:")
    print(f"Dimensions: {x_size} rows × {y_size} columns")
    


def main():
    # Example usage
    print("Enter your warehouse layout (press Enter twice when done):")
    print("Use:")
    print("  'e' for endpoints (blue tiles)")
    print("  '@' for obstacles (black tiles)")
    print("  'w' for workstations")
    print("  '.' for empty spaces")
    print("\nExample layout:")
    print("..@.w\n.e.e.\n..@.w")
    print("\nEnter your layout:")
    
    # Read input until empty line
    layout_lines = []
    while True:
        line = input()
        if not line:
            break
        layout_lines.append(line)
    
    layout_string = '\n'.join(layout_lines)
    
    # Convert to grid and validate
    grid = string_to_grid(layout_string)
    result = validate_blue_connectivity(grid)
    
    # Print results
    print("\nValidation Results:")
    print(f"Grid dimensions: {result['dimensions'][0]}x{result['dimensions'][1]}")
    print("\nTile counts:")
    for tile, count in result['tile_counts'].items():
        tile_name = {
            'e': 'Endpoints',
            '@': 'Obstacles',
            'w': 'Workstations',
            '.': 'Empty spaces'
        }[tile]
        print(f"{tile_name}: {count}")
    
    print(f"\nVisited endpoints: {result['visited_blue_tiles']}")
    print(f"Start position: {result['start_position']}")
    print(f"All endpoints connected: {result['is_connected']}")
    
    if not result['is_connected']:
        print("\nUnconnected endpoints at positions:")
        for pos in result['unvisited_blue_tiles']:
            print(f"Row {pos[0]}, Column {pos[1]}")
    
    # Print detailed grid statistics
    print_grid_stats(grid)

if __name__ == "__main__":
    main() 