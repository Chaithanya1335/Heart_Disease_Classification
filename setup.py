from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def getrequirements(path:str)->List[str]:
    '''
    This function return requiremets in list
    '''
    requirements = []
    with open(path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace('/n',"") for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements




setup(
    name= 'heart_disease_classification',
    version='0.0.1',
    author='GnanaChaithanya',
    author_email='m.gnanachaithanya12@gmail.com',
    packages= find_packages(),
    install_requires = getrequirements('Requirements.txt'))