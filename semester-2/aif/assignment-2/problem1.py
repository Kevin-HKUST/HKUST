import heapq
import argparse
import sys

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def read_puzzle(filename):
    with open(filename, 'r') as f:
        nums = []
        for line in f:
            nums.extend(int(x) for x in line.split())
    return tuple(nums)


def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    row, col = idx // 3, idx % 3
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in moves:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            nidx = nr * 3 + nc
            new_state = list(state)
            new_state[idx], new_state[nidx] = new_state[nidx], new_state[idx]
            neighbors.append(tuple(new_state))
    return neighbors


def h0(state):
    """Misplaced tiles heuristic."""
    count = 0
    for i in range(9):
        if state[i] != 0 and state[i] != GOAL_STATE[i]:
            count += 1
    return count


def h1(state):
    """Manhattan distance heuristic."""
    distance = 0
    for i in range(9):
        if state[i] != 0:
            goal_idx = GOAL_STATE.index(state[i])
            curr_row, curr_col = i // 3, i % 3
            goal_row, goal_col = goal_idx // 3, goal_idx % 3
            distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
    return distance


def h2(state):
    """Manhattan distance + linear conflict heuristic."""
    distance = h1(state)
    conflict = 0

    for row in range(3):
        for col1 in range(3):
            for col2 in range(col1 + 1, 3):
                idx1 = row * 3 + col1
                idx2 = row * 3 + col2
                val1, val2 = state[idx1], state[idx2]
                if val1 == 0 or val2 == 0:
                    continue
                goal_idx1 = GOAL_STATE.index(val1)
                goal_idx2 = GOAL_STATE.index(val2)
                if goal_idx1 // 3 == row and goal_idx2 // 3 == row:
                    if goal_idx1 % 3 > goal_idx2 % 3:
                        conflict += 1

    for col in range(3):
        for row1 in range(3):
            for row2 in range(row1 + 1, 3):
                idx1 = row1 * 3 + col
                idx2 = row2 * 3 + col
                val1, val2 = state[idx1], state[idx2]
                if val1 == 0 or val2 == 0:
                    continue
                goal_idx1 = GOAL_STATE.index(val1)
                goal_idx2 = GOAL_STATE.index(val2)
                if goal_idx1 % 3 == col and goal_idx2 % 3 == col:
                    if goal_idx1 // 3 > goal_idx2 // 3:
                        conflict += 1

    return distance + 2 * conflict


def a_star(start, heuristic_fn):
    """A* search returning (path, nodes_expanded)."""
    counter = 0
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start), counter, start, [start]))
    closed_set = set()
    nodes_expanded = 0

    while open_set:
        f, _, current, path = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_expanded += 1

        if current == GOAL_STATE:
            return path, nodes_expanded

        for neighbor in get_neighbors(current):
            if neighbor not in closed_set:
                g = len(path)
                h = heuristic_fn(neighbor)
                counter += 1
                heapq.heappush(open_set, (g + h, counter, neighbor, path + [neighbor]))

    return None, nodes_expanded


def format_state(state):
    lines = []
    for i in range(3):
        lines.append(' '.join(str(state[i * 3 + j]) for j in range(3)))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='8-Puzzle Solver using A* Search')
    parser.add_argument('-p', '--puzzle', type=str, help='Puzzle file path')
    parser.add_argument('-H', '--heuristic', type=int, choices=[0, 1, 2], default=1,
                        help='0=misplaced, 1=manhattan, 2=manhattan+linear_conflict')
    parser.add_argument('-o', '--output', type=str, help='Output file path')
    parser.add_argument('--batch', action='store_true', help='Run all puzzles with all heuristics')
    args = parser.parse_args()

    heuristics = {
        0: ('Misplaced Tiles', h0),
        1: ('Manhattan Distance', h1),
        2: ('Manhattan + Linear Conflict', h2),
    }

    if args.batch:
        print(f"{'Puzzle':<12} {'Heuristic':<30} {'Steps':<8} {'Nodes Expanded':<15}")
        print('-' * 65)
        for i in range(1, 6):
            filename = f'puzzle_{i}.txt'
            try:
                start = read_puzzle(filename)
            except FileNotFoundError:
                print(f"File {filename} not found, skipping.")
                continue
            for h_id in [0, 1, 2]:
                name, fn = heuristics[h_id]
                path, expanded = a_star(start, fn)
                steps = len(path) - 1 if path else -1
                print(f"puzzle_{i:<6} {name:<30} {steps:<8} {expanded:<15}")
        return

    if not args.puzzle:
        print("Please specify a puzzle file with -p or use --batch mode.")
        sys.exit(1)

    start = read_puzzle(args.puzzle)
    name, fn = heuristics[args.heuristic]

    print(f"Solving with heuristic: {name}")
    path, expanded = a_star(start, fn)

    if path is None:
        print("No solution found.")
    else:
        print(f"Solution found in {len(path) - 1} steps.")
        print(f"Nodes expanded: {expanded}")

        if args.output:
            with open(args.output, 'w') as f:
                for state in path:
                    f.write(format_state(state))
                    f.write('\n\n')
            print(f"Solution path written to {args.output}")


if __name__ == '__main__':
    main()