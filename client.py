import requests
from threading import Thread, Lock
import pdb

def looper(port,lock):
    for index in range(100):
        requests.post(f'http://localhost:{port}/transaction/new',json={'sender':f'me{index}', 'recipient':f'you{index}', 'code':f'port{port}'})
    lock.release()
        
if __name__ == '__main__':
#    pdb.set_trace()
#    r=requests.post('http://localhost:5000/transaction/new',json={'sender':'me', 'recipient':'you', 'code':'shalom'})
#    print(r.json())
#    r=requests.post('http://localhost:5000/transaction/new',json={'sender':'him', 'recipient':'her', 'code':'goodbye'})
#    print(r.json())
    lock0 = Lock()
    lock1 = Lock()
    lock2 = Lock()
    lock0.acquire()
    lock1.acquire()
    lock2.acquire()
    Thread(target=looper,args=(5000,lock0),daemon=True).start()
    Thread(target=looper,args=(5001,lock1),daemon=True).start()
    Thread(target=looper,args=(5002,lock2),daemon=True).start()
    lock0.acquire()
    lock1.acquire()
    lock2.acquire()
    r=requests.get('http://localhost:5000/transaction/get',json={'index':0})
    print(r.json())
    r=requests.get('http://localhost:5001/transaction/get',json={'index':0})
    print(r.json())
    r=requests.get('http://localhost:5002/transaction/get',json={'index':0})
    print(r.json())
