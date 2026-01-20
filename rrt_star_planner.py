# rrt_star_planner.py
# --- NEW FILE ---
# Contains the core 3D RRT* pathfinding algorithm.

import numpy as np

class Node:
    """A node in the RRT* tree."""
    def __init__(self, pos):
        self.pos = np.array(pos) # (x, y, z)
        self.parent = None
        self.cost = 0.0 # Cost from start node

class RRTStar:
    def __init__(self, start, goal, obstacles, bounds, 
                 step_size=30.0, search_radius=50.0, n_iterations=500):
        self.start_node = Node(start)
        self.goal_node = Node(goal)
        self.obstacles = obstacles
        self.bounds_min = np.array([bounds[0], bounds[2], bounds[4]])
        self.bounds_max = np.array([bounds[1], bounds[3], bounds[5]])
        self.step_size = step_size
        self.search_radius = search_radius
        self.n_iterations = n_iterations
        self.tree = [self.start_node]

    def run(self):
        """Runs the RRT* algorithm."""
        for _ in range(self.n_iterations):
            random_pos = self._get_random_pos()
            nearest_node = self._get_nearest_node(random_pos)
            new_pos = self._steer(nearest_node.pos, random_pos)
            
            if not self._is_collision_free(nearest_node.pos, new_pos):
                continue
                
            new_node = Node(new_pos)
            
            # Find the best parent in the search radius (the "star" part)
            near_nodes = self._find_near_nodes(new_node)
            self._choose_parent(new_node, near_nodes)
            
            self.tree.append(new_node)
            
            # Rewire the tree
            self._rewire(new_node, near_nodes)
            
        return self._get_final_path()

    def _get_random_pos(self):
        """Gets a random position within the bounds."""
        # 10% chance to sample the goal
        if np.random.rand() < 0.1:
            return self.goal_node.pos
        return np.random.rand(3) * (self.bounds_max - self.bounds_min) + self.bounds_min

    def _get_nearest_node(self, pos):
        """Finds the nearest node in the tree to a given position."""
        distances = [np.linalg.norm(node.pos - pos) for node in self.tree]
        return self.tree[np.argmin(distances)]

    def _steer(self, from_pos, to_pos):
        """Moves a 'step_size' from 'from_pos' towards 'to_pos'."""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return to_pos
        return from_pos + (direction / dist) * self.step_size

    def _is_collision_free(self, from_pos, to_pos):
        """Checks if the line segment between two positions hits an obstacle."""
        for obs in self.obstacles:
            obs_pos = np.array([obs['x'], obs['y'], obs['z']])
            safety_radius = 30.0 + obs.get('radius', 20.0)
            
            # Simple line-obstacle collision check
            line_vec = to_pos - from_pos
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0.0:
                return np.linalg.norm(from_pos - obs_pos) > safety_radius

            obs_vec = obs_pos - from_pos
            t = np.dot(obs_vec, line_vec) / line_len_sq
            
            if t < 0.0:
                closest_point = from_pos
            elif t > 1.0:
                closest_point = to_pos
            else:
                closest_point = from_pos + t * line_vec
                
            if np.linalg.norm(closest_point - obs_pos) <= safety_radius:
                return False
        return True

    def _find_near_nodes(self, new_node):
        """Finds all nodes within the search_radius of the new node."""
        near_nodes = []
        for node in self.tree:
            if np.linalg.norm(node.pos - new_node.pos) < self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def _choose_parent(self, new_node, near_nodes):
        """Finds the best parent for new_node from the near_nodes list."""
        best_node = self.tree[0] # Start with nearest
        if near_nodes:
            best_node = near_nodes[0]

        min_cost = self._get_cost(best_node) + np.linalg.norm(best_node.pos - new_node.pos)
        new_node.parent = best_node
        new_node.cost = min_cost

        for node in near_nodes:
            cost = self._get_cost(node) + np.linalg.norm(node.pos - new_node.pos)
            if cost < min_cost and self._is_collision_free(node.pos, new_node.pos):
                min_cost = cost
                new_node.parent = node
                new_node.cost = cost

    def _rewire(self, new_node, near_nodes):
        """Rewires the tree, checking if new_node provides a cheaper path for its neighbors."""
        for node in near_nodes:
            if node == new_node.parent:
                continue
            
            cost = self._get_cost(new_node) + np.linalg.norm(new_node.pos - node.pos)
            if cost < self._get_cost(node) and self._is_collision_free(new_node.pos, node.pos):
                node.parent = new_node
                node.cost = cost

    def _get_cost(self, node):
        return node.cost

    def _get_final_path(self):
        """Backtracks from the goal to get the final path."""
        # Find the node in the tree closest to the goal
        goal_node = self._get_nearest_node(self.goal_node.pos)
        
        # Check if goal is reachable (within one step)
        if np.linalg.norm(goal_node.pos - self.goal_node.pos) > self.step_size:
            return None # No path found
            
        path = [self.goal_node.pos]
        current = goal_node
        while current.parent is not None:
            path.append(current.pos)
            current = current.parent
        path.append(self.start_node.pos)
        return np.array(path[::-1]) # Return from start to goal
