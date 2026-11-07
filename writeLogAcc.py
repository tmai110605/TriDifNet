# -*- coding: utf-8 -*-
"""
Log writing utility for TriDiffNet training and validation accuracy.
"""
import os.path
from datetime import datetime

def writeLogAcc(filename='LogAcc1.txt', strtext=''):
    if not os.path.exists(filename):
        file1 = open(filename, "w")
    else:
        file1 = open(filename, "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    strtext = dt_string + ' ' + strtext + '\n'
    file1.writelines(strtext)
    file1.close()

