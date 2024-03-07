#!/usr/bin/env python3
"""
Uses the GitHub API to print the location of a specific user,
where user is passed as first argument of the script with full API URL

ex) "./2-user_location.py https://api.github.com/users/holbertonschool"
"""

from sys import argv
from time import time
import requests 

if __name__ == "__main__":
    if len(argv) < 2:
        raise TypeError(
            "Input must have the full API URL passed in as an argument: {}{}".
            format('ex. "./2-user_location.py',
                   'https://api.github.com/users/holbertonschool"'))
    try:
        url = argv[1]
        results = requests.get('url')
        if results.status_code == 403:
            reset = results.headers.get('X-Ratelimit-Reset')
            wait_time = int(reset) - time()
            minutes = round(wait_time/60)
            print('Reset in {} min'.format(minutes))
        else:
            results = results.json()
            location = results.get('location')
            if location:
                print(location)
            else:
                print('Not found')
    except Exception as err:
        print('Not found')
