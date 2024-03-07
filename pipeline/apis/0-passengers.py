#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and return the list of ships
that can hold a given number of passengers
"""

import requests


def availableShips(passengerCount):
    '''
    This function returns a list of starships that
      can hold a given number of passengers.

    Parameters:
    passengerCount (int): The minimum number
      of passengers a starship must be able to hold.

    Returns:
    shipslist (list): A list of starship
      names that can hold at least 'passengerCount' passengers.
    '''

    # API endpoint to fetch starship data
    url = "https://swapi-api.alx-tools.com/api/starships/?format=json"

    # List to store all starship data
    ships = []

    # Loop until all pages of the API response have been processed
    while url:
        # Get the API response
        response = requests.get(url).json()

        # Add the results from the current page to our list
        ships += response.get('results')

        # Get the URL for the next page
        url = response.get('next')

    shipslist = []

    # Loop through each starship
    for ship in ships:
        # Get the passenger capacity, removing any commas
        passengers = ship.get('passengers').replace(",", "")

        if passengers != "n/a" and passengers != 'unknown':
            if int(passengers) >= passengerCount:
                shipslist.append(ship.get('name'))

    # Return the list of suitable starships
    return shipslist
