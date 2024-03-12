#!/usr/bin/env python3
"""Defines a function that returns the list of schools
that have a specific topic"""
def schools_by_topic(mongo_collection, topic):
    """args: mongo_collection: collection to search in
            topic: topic to search for
            return: list of schools with topic"""
    schools = []
    collection = mongo_collection.find({'topics': topic})
    for document in collection:
        schools.append(document)
    return schools
