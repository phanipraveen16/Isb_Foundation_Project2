import json
import requests

def api_call(path,steps):
    d = {'steps': steps} 
    s = json.dumps(d)
    header = {'Content-Type': 'application/json', \
              'Accept': 'application/json'}
    resp = requests.post(path, data=s, headers=header)
    print("resp is",resp)
    output = resp.json()
    return output

path = "https://isb-foundation-project2.herokuapp.com/forecast"
steps = 4
print(api_call(path,steps))

