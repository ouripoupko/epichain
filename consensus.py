
from blockchain import Blockchain
from state import StateMachine
from network import Network
from block import Block

import pdb

class Consensus:

    def __init__(self):
        self.state = StateMachine()
        self.blockchain = Blockchain()
        self.transaction_pool = []
        self.nodes = []

    def set_port(self, port):
        ports = [5000, 5001, 5002]
        for p in ports:
            self.nodes.append(f'127.0.0.1:{p}')
        self.me = f'127.0.0.1:{port}'
        neighbours = list(self.nodes)
        neighbours.remove(self.me)
        self.network = Network(self.me, neighbours)
        
    def new_transaction(self, transaction):
        # drive consensus over the transaction
        transaction.set_registrer(self.me)
        self.add_transaction(transaction)
        return True

    def report_transaction(self, transaction):
        self.add_transaction(transaction)

    def add_transaction(self, transaction):
        if transaction not in self.transaction_pool:
            self.transaction_pool.append(transaction)
            print(self.transaction_pool)
            self.network.broadcast_transaction(transaction)
            if self.my_turn():
                self.sign()

    def my_turn(self):
        last_signer = self.blockchain.last_signer()
        last_index = self.nodes.index(last_signer) if last_signer else -1
        next = 0 if last_index+1 == len(self.nodes) else last_index+1
        return self.nodes[next] == self.me
        
    def sign(self):
        block = Block(self.transaction_pool, self.me, self.blockchain.last_hash())
        self.blockchain.add(block)
        self_transaction_pool = []
        self.network.broadcast_block(block)
        
    def report_block(self, transactions, signer, previous):
        if self.blockchain.last_hash() == previous:
            block = Block(self.transaction_pool, signer, previous)
            self.blockchain.add(block)
            for transaction in transactions:
                if transaction in self.transaction_pool:
                    self.transaction_pool.remove(transaction)
            self.network.broadcast_block(block)
