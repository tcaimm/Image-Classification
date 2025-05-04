import sys
import os

# Insert project root directory into sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.testers.inference import main

if __name__ == '__main__':
    main()