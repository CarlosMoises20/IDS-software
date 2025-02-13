
from pyspark.sql.types import *
from pyspark.sql.functions import expr, struct, explode, col
from pyspark.ml.feature import Imputer                  # includes 'mean', 'median' and 'mode'
#from sklearn.impute import KNNImputer, SimpleImputer    # SimpleImputer includes 'mean', 'median', 'mode' and 'constant'
from pyspark.ml.regression import LinearRegression

"""
Auxiliary function to indicate if type of object 'obj' belongs to one of the 
instances inside the array 'instances'. If so, it returns True. Otherwise, it
returns False. 

"""
def is_one_of_instances(obj, instances):
    
    for instance in instances:
        if isinstance(obj, instance):
            return True
        
    return False



"""
Auxiliary function to get all numeric colums of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_numeric_attributes(schema, parent_name=""):
    
    numeric_attributes = []

    # Iterate through all the fields in the schema, including fields inside arrays and structs
    for field in schema.fields:
        field_name = f"{parent_name}.{field.name}" if parent_name else field.name  # Handle nested fields
        
        if is_one_of_instances(field.dataType, [DoubleType, FloatType, IntegerType, LongType, ShortType]):
            numeric_attributes.append(field_name)
        
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            numeric_attributes.extend(get_numeric_attributes(field.dataType.elementType, field_name))  # Recursive call for nested structs
        
        elif isinstance(field.dataType, StructType):
            numeric_attributes.extend(get_numeric_attributes(field.dataType, field_name))  # Handle direct nested structs
    
    return numeric_attributes


"""
Auxiliary function to get all string colums of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_string_attributes(schema, parent_name=""):
    
    string_attributes = []

    # Iterate through all the fields in the schema, including fields inside arrays and structs
    for field in schema.fields:
        field_name = f"{parent_name}.{field.name}" if parent_name else field.name  # Handle nested fields
        
        if isinstance(field.dataType, StringType):
            string_attributes.append(field_name)
        
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            string_attributes.extend(get_string_attributes(field.dataType.elementType, field_name))  # Recursive call for nested structs
        
        elif isinstance(field.dataType, StructType):
            string_attributes.extend(get_string_attributes(field.dataType, field_name))  # Handle direct nested structs
    
    return string_attributes



"""
Auxiliary function to get all boolean colums of a spark dataframe schema

    schema - spark dataframe schema
    parent_name - name of parent field of an array or struct. Optional, only applicable of the field is array or struct and
                    used for recursive calls inside the function.

Returns: a array

"""
def get_boolean_attributes(schema, parent_name=""):
    
    boolean_attributes = []

    # Iterate through all the fields in the schema, including fields inside arrays and structs
    for field in schema.fields:
        field_name = f"{parent_name}.{field.name}" if parent_name else field.name  # Handle nested fields
        
        if isinstance(field.dataType, BooleanType):
            boolean_attributes.append(field_name)
        
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            boolean_attributes.extend(get_boolean_attributes(field.dataType.elementType, field_name))  # Recursive call for nested structs
        
        elif isinstance(field.dataType, StructType):
            boolean_attributes.extend(get_boolean_attributes(field.dataType, field_name))  # Handle direct nested structs
    
    return boolean_attributes



"""
This function applies pre-processing on data from the dataframe 'df', for the 'rxpk' dataset

 1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
        selecting only the attributes that are relevant to build the intended model_numeric for IDS 

 2 - Imputes missing values

"""
def pre_process_rxpk_dataset(df):

    ### FEATURE SELECTION: remove irrelevant / redundant attributes
    df = df.drop("type", "totalrxpk", "fromip")

    # apply filter to let pass only relevant attributes inside 'rxpk' array
    df = df.withColumn("rxpk", expr("transform(rxpk, x -> named_struct( 'AppEUI', x.AppEUI, \
                                    'AppNonce', x.AppNonce, \
                                    'DLSettings', x.DLSettings, \
                                    'DLSettingsRX1DRoffset', x.DLSettingsRX1DRoffset, \
                                    'DLSettingsRX2DataRate', x.DLSettingsRX2DataRate, \
                                    'DevAddr', x.DevAddr, \
                                    'DevEUI', x.DevEUI, \
                                    'DevNonce', x.DevNonce, \
                                    'Direction', x.Direction, \
                                    'FCnt', x.FCnt, \
                                    'FCtrl', x.FCtrl, \
                                    'FCtrlACK', x.FCtrlACK, \
                                    'FCtrlADR', x.FCtrlADR, \
                                    'FHDR', x.FHDR, \
                                    'FOpts', x.FOpts, \
                                    'FPort', x.FPort, \
                                    'FRMPayload', x.FRMPayload, \
                                    'MACPayload', x.MACPayload, \
                                    'MHDR', x.MHDR, \
                                    'MIC', x.MIC, \
                                    'MessageType', x.MessageType, \
                                    'NetID', x.NetID, \
                                    'PHYPayload', x.PHYPayload, \
                                    'RxDelay', x.RxDelay, \
                                    'RxDelayDel', x.RxDelayDel, \
                                    'chan', x.chan, \
                                    'codr', x.codr, \
                                    'data', x.data, \
                                    'datr', x.datr, \
                                    'freq', x.freq, \
                                    'lsnr', x.lsnr, \
                                    'rfch', x.rfch, \
                                    'rsig', transform(x.rsig, rs -> named_struct( \
                                                    'chan', rs.chan, \
                                                    'lsnr', rs.lsnr \
                                    )), \
                                    'rssi', x.rssi, \
                                    'size', x.size, \
                                    'time', x.time, \
                                    'tmms', x.tmms, \
                                    'tmst', x.tmst ))")
                    )


    #numeric_attributes = get_numeric_attributes(df.schema)
    #string_attributes = get_string_attributes(df.schema)
    #boolean_attributes = get_boolean_attributes(df.schema)

    
    
    # TODO: impute missing values for numeric attributes: using the 'mean' strategy
    #mv_imputer_numeric = SimpleImputer(missing_values=None, strategy='mean')
    mv_imputer = Imputer(missing_values=None, strategy='mean')
    
    """
    mv_imputer.fit_transform(df).show()

    #df_exploded = df.withColumn("rxpk", explode(col("rxpk")))
    #model_numeric = mv_imputer_numeric.fit(df_exploded.select("rxpk.freq", "rxpk.rssi", "rxpk.DLSettingsRX2DataRate")) 
    #model_numeric.transform(df).show()
    
    """

    # TODO: impute missing values for string & categorical attributes: analyse the best strategy (most frequent??)

    
    return df




"""

This function applies pre-processing on data from the dataframe 'df', for the 'txpk' dataset

 1 - Applies feature selection techniques to remove irrelevant attributes (dimensionality reduction),
        selecting only the attributes that are relevant to build the intended model_numeric for IDS 

 2 - Imputes missing values

"""
def pre_process_txpk_dataset(df):

    ## Feature Selection: remove irrelevant / redundant attributes
    df = df.drop("type")

    # apply filter to let pass only relevant attributes inside 'txpk' 
    df = df.withColumn("txpk", struct(
                           expr("txpk.data AS data"),
                           expr("txpk.datr AS datr"),
                           expr("txpk.freq AS freq"),
                           expr("txpk.powe AS powe"),
                           expr("txpk.size AS size"),
                           expr("txpk.tmst AS tmst")
                        ))


    return df

