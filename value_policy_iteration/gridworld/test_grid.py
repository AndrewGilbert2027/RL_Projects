import unittest
from grid import GridWorld

class TestGridWorld(unittest.TestCase):
    def setUp(self):
        # Default map for testing
        self.default_map = [
            ['s', '.', '.', '.'],
            ['h', '.', '.', 'h'],
            ['.', '.', '.', '.'],
            ['.', 'h', '.', 'g']
        ]
        self.grid = GridWorld(map=self.default_map)

    def test_get_start(self):
        self.assertEqual(self.grid.get_start(), (0, 0))  # Start position should be (0, 0)

    def test_get_goal(self):
        self.assertEqual(self.grid.get_goal(), (3, 3))  # Goal position should be (3, 3)

    def test_get_holes(self):
        expected_holes = [(1, 0), (1, 3), (3, 1)]
        self.assertEqual(self.grid.get_holes(), expected_holes)  # Check hole positions

    def test_valid_holes(self):
        self.assertTrue(self.grid._GridWorld__valid_holes())  # Ensure holes are valid

    def test_neighbors(self):
        neighbors = self.grid._GridWorld__get_neighbors((0, 0))
        expected_neighbors = [(1, 0), (0, 1)]  # Neighbors of (0, 0) should be (0, 1) and (1, 0)
        self.assertEqual(neighbors, expected_neighbors)  # Check neighbors of (0, 0)

    def test_invalid_position(self):
        self.assertFalse(self.grid._GridWorld__is_valid((-1, 0)))  # Out of bounds
        self.assertTrue(self.grid._GridWorld__is_valid((0, 1)))    # Valid position

if __name__ == '__main__':
    unittest.main()