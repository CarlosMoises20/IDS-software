
import sys
import os
from crate.client import connect

# Move back to the root directory to access the constants file in order to apply the import with success
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import CRATEDB_URI

# Connect to CrateDB database using the URI and the method from crate client library 
db_connect = connect(CRATEDB_URI)
cursor = db_connect.cursor()

cursor.execute("DELETE FROM TABLE IF EXISTS model")
cursor.execute("DELETE FROM TABLE IF EXISTS frame_counter")