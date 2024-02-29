import sys
import argparse
from streamlit.web.cli import main

if __name__ == '__main__':
    args = sys.argv.copy()
    
    sys.argv = ["streamlit", "run", "src/streamlit.py", "--"]
    sys.argv.extend(args[1:])
    
    sys.exit(main())