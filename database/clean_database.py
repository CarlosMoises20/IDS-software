
from crate.client import connect
from constants import CRATEDB_URI

db_connect = connect(CRATEDB_URI)

cursor = db_connect.cursor()

cursor.execute("DELETE FROM TABLE IF EXISTS model")
cursor.execute("DELETE FROM TABLE IF EXISTS frame_counter")