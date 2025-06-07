from datetime import datetime
start_time = str(int(datetime(2025, 4, 1).timestamp()*1000))
end_time = str(int(datetime(2025, 5, 1).timestamp()*1000))

print(start_time, end_time)