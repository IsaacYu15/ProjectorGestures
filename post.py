import requests

url = 'http://localhost:80/elm/groups/Group01/performer?active=1&sequenceId=2'
myobj = {'somekey': 'somevalue'}

x = requests.post(url)
