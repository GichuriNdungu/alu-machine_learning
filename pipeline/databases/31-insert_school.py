#!/usr/bin/env python3
"""Defines a function that inserts a new document to a collection"""


def insert_school(mongo_collection, **kwargs):
    '''args: mongo_collection: collection to add to
            **kwargs: documents to add
            return inserted_id'''

    doc = mongo_collection.insert_one(kwargs)
    return (doc.inserted_id)
