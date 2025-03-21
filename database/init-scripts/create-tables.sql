
DROP TABLE IF EXISTS model;
DROP TABLE IF EXISTS frame_counter;

CREATE TABLE model (
    dev_addr            VARCHAR(8) NOT NULL,
    dataset_type        VARCHAR(4) NOT NULL CHECK (dataset_type IN ('rxpk', 'txpk')),
    accuracy            DECIMAL(5,4) NOT NULL CHECK (accuracy >= 0 AND accuracy <= 1),
    created             TIMESTAMP NOT NULL DEFAULT NOW(),
    modified            TIMESTAMP NOT NULL DEFAULT NOW(),
    --model               OBJECT(DYNAMIC) NOT NULL,     -- review this later
    
    CONSTRAINT PK_model PRIMARY KEY (dev_addr, dataset_type)
);

CREATE TABLE frame_counter(
    dev_addr            VARCHAR(8) NOT NULL,
    dataset_type        VARCHAR(4) NOT NULL CHECK (dataset_type IN ('rxpk', 'txpk')),
    created             TIMESTAMP NOT NULL DEFAULT NOW(),
    modified            TIMESTAMP NOT NULL DEFAULT NOW(),
    fcnt                INT NOT NULL,
    
    CONSTRAINT PK_frame_counter PRIMARY KEY (dev_addr)
);