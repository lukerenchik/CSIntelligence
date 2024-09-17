import unittest
from MatchGuesserSite.MatchGuesser import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_register_and_login(self):
        # Register a new user
        response = self.app.post('/api/register', json={
            'name': 'testuser',
            'pass': 'testpassword'
        })
        self.assertEqual(response.status_code, 201)

        # Log in with the new user
        response = self.app.post('/api/login', json={
            'name': 'testuser',
            'pass': 'testpassword'
        })
        self.assertEqual(response.status_code, 200)

    def test_increment_counters(self):
        # Log in first
        self.app.post('/api/login', json={
            'name': 'testuser',
            'pass': 'testpassword'
        })

        # Get initial counters
        response = self.app.get('/api/get_user_counters')
        data = response.get_json()
        correct_before = data.get('correct_guesses', 0)
        incorrect_before = data.get('incorrect_guesses', 0)

        # Increment correct guesses
        self.app.post('/api/increment_correct')
        # Increment incorrect guesses
        self.app.post('/api/increment_incorrect')

        # Get updated counters
        response = self.app.get('/api/get_user_counters')
        data = response.get_json()
        correct_after = data.get('correct_guesses', 0)
        incorrect_after = data.get('incorrect_guesses', 0)

        # Assert the counters have incremented
        self.assertEqual(correct_after, correct_before + 1)
        self.assertEqual(incorrect_after, incorrect_before + 1)

if __name__ == '__main__':
    unittest.main()
