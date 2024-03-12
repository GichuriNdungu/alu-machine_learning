#!/usr/bin/env python3
"""Defines a function that changes all
 topics of a school document based on the name"""
def  update_topics(mongo_collection, name, topics):
    """params; 
        mongo_collection: collection to update
        name: name of school to update
        topics: topics to update"""
    mongo_collection.update_many({'name': name}, 
                                 {'$set': 'topics': topics})