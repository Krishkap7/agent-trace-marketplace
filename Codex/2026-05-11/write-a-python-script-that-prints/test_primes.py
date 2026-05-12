import io
import unittest
from contextlib import redirect_stdout

import primes


class PrimeScriptTests(unittest.TestCase):
    def test_first_ten_primes(self):
        self.assertEqual(primes.first_primes(10), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

    def test_main_prints_first_ten_primes(self):
        output = io.StringIO()

        with redirect_stdout(output):
            primes.main()

        self.assertEqual(output.getvalue(), "2\n3\n5\n7\n11\n13\n17\n19\n23\n29\n")


if __name__ == "__main__":
    unittest.main()
