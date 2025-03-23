
import sys
import os
from crate.client import connect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import CRATEDB_URI

db_connect = connect(CRATEDB_URI)

cursor = db_connect.cursor()

cursor.execute("DELETE FROM TABLE IF EXISTS model")
cursor.execute("DELETE FROM TABLE IF EXISTS frame_counter")