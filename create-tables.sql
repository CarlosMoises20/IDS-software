


CREATE TYPE DatasetType AS ENUM ('RXPK', 'TXPK');


CREATE TABLE model (
    dev_addr        BIGINT NOT NULL,
    dataset_type    DatasetType NOT NULL,
    created         TIMESTAMP NOT NULL DEFAULT NOW(),
    modified        TIMESTAMP NOT NULL DEFAULT NOW(),
    model           JSONB NOT NULL,
    accuracy        DECIMAL(5,4) NOT NULL CHECK (accuracy >= 0 AND accuracy <= 1),
    
    CONSTRAINT PK_model PRIMARY KEY (dev_addr, dataset_type)
);


CREATE TABLE frame_counter(
    dev_addr            BIGINT NOT NULL,
    dataset_type        DatasetType NOT NULL,
    created             TIMESTAMP NOT NULL DEFAULT NOW(),
    modified            TIMESTAMP NOT NULL DEFAULT NOW(),
    fcnt                INT NOT NULL,
    
    CONSTRAINT PK_frame_counter PRIMARY KEY (dev_addr, dataset_type)
)