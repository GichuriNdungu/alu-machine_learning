#!/usr/bin/env python3
"""Defines  a script that prints
 the location of a specific github user"""

from sys import argv
from time import time
import requests 

if __name__ == "__main__":

    try:
        url = argv[1]
        results = requests.get('url')
        if results.status_code == 403:
            reset = results.headers.get('X-Ratelimit-Reset')
            wait_time = int(reset) - time()
            minutes = round(wait_time/60)
            print(f'reset in {minutes} min')
        else:
            results = results.json()
            location = results.get('location')
            if location:
                print(location)
            else:
                print('Not found')
    except Exception as err:
        print('Not found')
