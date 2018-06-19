import requests
import pdb

if __name__ == '__main__':
    pdb.set_trace()
    r=requests.post('http://localhost:5000/transaction/new',json={'sender':'me', 'recipient':'you', 'code':'shalom'})
    print(r.json())
    r=requests.post('http://localhost:5000/transaction/new',json={'sender':'him', 'recipient':'her', 'code':'goodbye'})
    print(r.json())
    r=requests.get('http://localhost:5000/transaction/get',json={'index':0})
    print(r.json())
    r=requests.get('http://localhost:5000/transaction/get',json={'index':1})
    print(r.json())
