#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and returns the
list of names of the home planets of all sentient species
"""

import requests


def sentientPlanets():
    '''Args:
    return: list of sentient homelands'''
    url = "https://swapi-api.alx-tools.com/api/species/?format=json"
    species = []
    while url:
        results = requests.get(url).json()
        species += results.get('results')
        url = results.get('next')
    planets = []
    for specie in species:
        if specie.get('designation') == 'sentient' or\
                specie.get('classification') == 'sentient':
            planet_url = specie.get('homeworld')
            if planet_url:
                planet_data = requests.get(planet_url).json()
                planet_name = planet_data.get('name')
                planets.append(planet_name)
    return planets
