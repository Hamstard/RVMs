"""
Master script to run all unittests from. Note that some contain 
tdda's ReferenceTestCase subclasses. Reference files shipped
with this module may be outdated and possibly require manual 
execution of scripts with ReferenceTestCase subclasses. Relevant 
scripts: test_parsing.py

Eric Schmidt
e.schmidt@cantab.net
2017-10-15
"""

import test_linear_model
import unittest, sys

if __name__ == "__main__":
        
    # (reference) test cases for linear_model regression
    suites = test_linear_model.get_suite()

    # run joint suite
    suite = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner()
    results = runner.run(suite).wasSuccessful()
    sys.exit(not results)