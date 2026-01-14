import sys
import os
sys.path.append(os.path.abspath("../"))
from ChunkerMaster.hyper_paramter import hyper_parameter

if __name__ == "__main__":
    
    print(hyper_parameter.CHUNK_OVERLAP)