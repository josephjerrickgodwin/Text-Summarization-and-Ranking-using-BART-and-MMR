import shutil
import os

from subprocess import STDOUT, PIPE, Popen
from rouge_compute import *

def main():

    # Copy summary to rouge folder
    shutil.copyfile("output\\summary.txt", "rouge_compute\\files\\system\\Task1_reference1.txt")

    # Change working directory
    os.chdir("rouge_compute")

    # Execute command
    pipeline_java = Popen(['java', '-jar', 'rouge2-1.2.2.jar'])

    # Print Output
    print('-'*96)  
    print('\t\t\t\tROUGE MERTIX CALCULATION - V2-1.2.2')
    print('-'*96)
    print(pipeline_java.stdout)
    print('-'*96)