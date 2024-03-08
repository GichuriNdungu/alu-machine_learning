#!/usr/bin/env python3
''' script that displays the upcoming launch with these information:

Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
by pinging the unofficial SpaceX server'''
import requests
from datetime import datetime
if __name__ == '__main__':

    url = 'https://api.spacexdata.com/v5/launches/upcoming?format=json'
    launch_pad_url = 'https://api.spacexdata.com/v4/launchpads?format=json'
    rocket_names_url = 'https://api.spacexdata.com/v4/rockets/?format=json'
    # lauch pads
    launch_pads = []
    launch_pad_names = requests.get(launch_pad_url).json()
    launch_pads += launch_pad_names
    # print(f'these are the launch_pads {launch_pads}')
    # rocket_names
    rocket_names = []
    roc_name = requests.get(rocket_names_url).json()
    rocket_names += roc_name
    # print(f'these are the rocket_names {rocket_names}')
    # upcoming launches
    upcoming = []
    results = requests.get(url).json()
    upcoming += results
    launch_info = []
    launch_times = []
    for launch in upcoming:
        sorted_upcoming = {}
        launch_time = launch.get('date_utc')
        dt = datetime.strptime(launch_time, "%Y-%m-%dT%H:%M:%S.%f%z")
        unix_time = dt.timestamp()
        sorted_upcoming['launch_name'] = launch.get('name')
        sorted_upcoming['local_date'] = launch.get('date_local')
        sorted_upcoming['rocket'] = launch.get('rocket')
        sorted_upcoming['launch_pad'] = launch.get('launchpad')
        sorted_upcoming['locality'] = ''
        launch_times.append(unix_time)
        sorted_upcoming['launch_time'] = unix_time
        launch_info.append(sorted_upcoming)

    launch_times.sort()

    for launch in launch_info:
        if launch['launch_time'] == launch_times[0]:
            roc_id = launch['rocket']
            pad_id = launch['launch_pad']
            for roc, pad in zip(rocket_names, launch_pads):
                if roc.get('id') == roc_id:
                    launch['rocket'] = roc.get('name')
                if pad.get('id') == pad_id:
                    launch['launch_pad'] = pad.get('name')
                    launch['locality'] = pad.get('locality')
            print(launch['launch_name'], '(' + launch['local_date'] + ')',
                  launch['rocket'], launch['launch_pad'], '(' + launch['locality'] + ')')
            break
