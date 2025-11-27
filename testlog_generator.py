import time, random

normal = [
    '10.0.0.1 - - [27/Nov/2025] "GET /home HTTP/1.1" 200 500\n',
    '10.0.0.2 - - [27/Nov/2025] "POST /login HTTP/1.1" 200 450\n'
]

while True:
    with open("logs/train_logs.txt","a") as f:
        f.write(random.choice(normal))
    time.sleep(1)
#===================================