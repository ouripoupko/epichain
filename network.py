import requests

class Network:
    def __init__(self, notifier, neighbours):
        self.notifier = notifier
        self.neighbours = neighbours

    def broadcast_transaction(self, transaction):
        transaction.set_notifier(self.notifier)
        for node in self.neighbours:
            response = requests.post(f'http://{node}/network/report', json=transaction.__dict__)

    def broadcast_block(self, block):
        for node in self.neighbours:
            response = requests.post(f'http://{node}/network/sign', json=block.to_dict())


