import re

def parse_log(line):
    pattern = r'(\d+\.\d+\.\d+\.\d+).*?\[(.*?)\]\s+"(.*?)"\s+(\d+)\s+(\d+)'
    match = re.search(pattern, line)
    if match:
        return {
            "ip": match.group(1),
            "timestamp": match.group(2),
            "request": match.group(3),
            "status": int(match.group(4)),
            "size": int(match.group(5))
        }
    return None

def featurize(parsed):
    return [len(parsed["request"]), parsed["status"], parsed["size"]]
