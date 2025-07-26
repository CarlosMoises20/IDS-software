
import socket
from kafka import KafkaProducer

"""
Auxiliary function to print processing time on an adequate format, in hours, minutes and seconds,
or milisseconds if "seconds" is a decimal lower than 1

"""
def format_time(seconds):

    # Convert seconds to milisseconds (round to integer)
    milisseconds = round(seconds * 1000)

    # If the time is less than 1 second and arrendondated value of milisseconds is less than 1000 milisseconds
    # For example: seconds = 0.99999 -> milisseconds = 999.99 -> round(999.99) = 1000 and time will be printed in seconds (1s)
    if milisseconds < 1000:
        return f"{milisseconds} ms"      # Milisseconds

    # Round the value of seconds to integer
    seconds = round(seconds)                

    # If the number of seconds is between 1 and 59
    if seconds < 60:
        return f"{seconds} s"                   # Seconds
    
    # If the number of seconds is between 60 and 3599
    if seconds < 3600:
        minutes = seconds // 60                 # Minute as the integer part of the division of total by number of seconds in a minute
        secs = seconds % 60                     # Seconds as the integer part of the rest of the division of total by number of seconds in a minute
        
        if secs == 0:                           # If "secs" are multiple of 60
            return f"{minutes} min"             # Only minutes when "minutes" are multiples of 60
        
        return f"{minutes} min {secs} s"        # Format in minutes and seconds

    # If the number of seconds is 3600 or higher        
    hours = seconds // 3600                         # Hour as the integer part of the division of total by number of seconds in a hour
    minutes = (seconds % 3600) // 60                # Minutes as the rest of the division of total by number of seconds in a hour and integer part of division by number of seconds in a minute
    secs = seconds % 60                             # Seconds as the integer part of the rest of the division of total by number of seconds in a minute
    
    if secs == 0:                                   # If "secs" are multiple of 3600
        if minutes == 0:                            # If "minutes" are multiple of 60
            return f"{hours} h"                     # Only hours when "minutes" are multiple of 60 and "secs" are multiple of 3600
        
        return f"{hours} h {minutes} min"           # Only hours and minutes when "secs" are multiple of 60
        
    return f"{hours} h {minutes} min {secs} s"      # Format in minutes, hours and seconds


"""
Auxiliary function that will open a UDP socket that receives LoRaWAN messages coming from a LoRa
gateway, using port 5200 to receive those messages. When the UDP socket receives those messages, it
will forward them to a Kafka producer that will have a topic called lorawan-messages, where the UDP socket
will send the received LoRaWAN messages. Those messages will, then, be available on the Kafka producer
on that topic, on port 9092, to be consumed by Spark, which will group the messages using batches
and process them 

"""
def udp_to_kafka_forwarder():

    UDP_IP = "0.0.0.0"
    UDP_PORT = 5200
    
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_IP, UDP_PORT))
    print(f"[*] Listening on UDP Port {UDP_PORT}")

    # Kafka setup
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: v.encode('utf-8')  # converte para bytes
    )

    while True:
        data, _ = udp_sock.recvfrom(4096)

        dec_data = data.decode(errors='ignore').strip()

        print(f"[UDP MESSAGE RECEIVED] {dec_data}")

        # Enviar para o tópico Kafka
        producer.send('lorawan-messages', dec_data)