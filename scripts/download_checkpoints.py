# Standard Library
import os, sys
import logging
from pathlib import Path

# MegaStream
from modules.utils.download import download_fastsam, download_megapose

if __name__ == '__main__':
    download_fastsam()
    download_megapose()