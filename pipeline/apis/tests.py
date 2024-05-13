#!/usr/bin/env python3
"""
Test file
"""
sentientPlanets = __import__('3-upcoming')
with open('3-upcoming', 'r') as f:
    contents = f.read()
    for char in contents:
        if ord(char) > 127:
            print(f'Non-ASCII character: {char}')