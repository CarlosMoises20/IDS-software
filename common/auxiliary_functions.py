

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

    else:
        # Function to round the value of seconds to integer
        seconds = round(seconds)                

        # If the number of seconds is between 1 and 59
        if seconds < 60:
            return f"{seconds} s"                   # Seconds
        
        # If the number of seconds is between 60 and 3599
        elif seconds < 3600:
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

