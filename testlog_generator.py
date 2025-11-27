import time, random

# normal = [
#     '10.0.0.1 - - [27/Nov/2025] "GET /home HTTP/1.1" 200 500\n',
#     '10.0.0.2 - - [27/Nov/2025] "POST /login HTTP/1.1" 200 450\n'
# ]

# while True:
#     with open("logs/train_logs.txt","a") as f:
#         f.write(random.choice(normal))
#     time.sleep(1)
#=================================== week2
import time, random

normal_logs = [
    '10.0.0.1 - - [27/Nov/2025] "GET /home HTTP/1.1" 200 500\n',
    '10.0.0.2 - - [27/Nov/2025] "POST /login HTTP/1.1" 200 450\n'
]

anomaly_logs = [
    '10.0.0.9 - - [27/Nov/2025] "GET /admin HTTP/1.1" 500 50\n'
]

while True:
    with open("logs/access.log","a") as f:
        f.write(random.choice(normal_logs) if random.random()>0.1 else random.choice(anomaly_logs))
    time.sleep(1)
