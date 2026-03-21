"""
Entry point: rebuild features and train the model.

Usage:
    python train_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import DB_PATH
from model.train import train_model

dc, xgb_model, importances = train_model(DB_PATH)
