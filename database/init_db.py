
import sys
import os

from crate.client import connect

# Move back to the root directory to access the constants file in order to apply the import with success
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.constants import CRATEDB_URI

# Connect to CrateDB database using the URI and the method from crate client library
db_connect = connect(CRATEDB_URI)
cursor = db_connect.cursor()

# SQL statement to create the 'model' table
create_model_table = """
            CREATE TABLE model (
                dev_addr            BIGINT NOT NULL,
                accuracy            DECIMAL(5,4) NOT NULL CHECK (accuracy >= 0 AND accuracy <= 1),
                created             TIMESTAMP NOT NULL DEFAULT NOW(),
                modified            TIMESTAMP NOT NULL DEFAULT NOW(),
                CONSTRAINT PK_model PRIMARY KEY (dev_addr, dataset_type)
            );
            """

# SQL statement to create the 'frame_counter' table
create_fcnt_table = """
                CREATE TABLE frame_counter (
                    dev_addr            BIGINT NOT NULL,
                    created             TIMESTAMP NOT NULL DEFAULT NOW(),
                    modified            TIMESTAMP NOT NULL DEFAULT NOW(),
                    fcnt                INT NOT NULL,
                    CONSTRAINT PK_frame_counter PRIMARY KEY (dev_addr)
                );
                """



# Execute table creation
try:
    cursor.execute("DROP TABLE IF EXISTS frame_counter")
    cursor.execute("DROP TABLE IF EXISTS model")
    cursor.execute(create_model_table)
    cursor.execute(create_fcnt_table)
    print("Tables created successfully.")

except Exception as e:
    print("Error creating tables:", e)
    
finally:
    cursor.close()
    db_connect.close()