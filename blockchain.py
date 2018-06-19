import hashlib
import json

# Creates a SHA-256 hash of a transaction
def hash(block):
    # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
    block_string = json.dumps(block.to_dict(), sort_keys=True).encode()
    hash_code = hashlib.sha256(block_string).hexdigest()
    return hash_code


class Blockchain:
    def __init__(self):
        self.blocks = []

    def new_transaction(self, sender, recipient, code):
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'code': code,
            'previous_hash': hash(self.transactions[-1]) if len(self.transactions) > 0 else None
        }

        self.transactions.append(transaction)

        return transaction

    def get_transaction(self, index):
        return self.transactions[index] if len(self.transactions)>0 else None

    def get_all(self):
        return self.blocks

    def last_signer(self):
        return self.blocks[-1].signer if self.blocks else None
        
    def add(self, block):
        self.blocks.append(block)

    def last_hash(self):
        if not self.blocks:
            return 0
        return hash(self.blocks[-1])
