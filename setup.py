#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup

setupArgs = {
    'name'         : 'パ MCTS',
    'version'      : '0.1',
    'description'  : 'Python library for Monte Carlo Tree Search',
    'author'       : 'Andrii Sevastianov',
    'author_email' : 'mustitz@gmail.com',
    'license'      : 'MIT',
    'packages'     : ['pa']
}

setup(**setupArgs)
