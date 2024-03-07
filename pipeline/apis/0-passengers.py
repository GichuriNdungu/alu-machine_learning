#!/usr/bin/env python3
import requests
def availableShips(passengerCount):
    '''returns a list of ships that can
      hold a given number of passengers'''
    url = "https://swapi-api.alx-tools.com/api/starships/?format=json"
    ships=[]
    while url:
        response = requests.get(url).json()
        ships += response.get('results')
        url = response.get('next')
    shipslist = []
    for ship in ships:
        passengers = ship.get('passengers').replace(",", "")
        if passengers != "n/a" and passengers !='unknown':
            if int(passengers) >= passengerCount:
                shipslist.append(ship.get('name'))
    return shipslist