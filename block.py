
class Block:
    def __init__(self, transactions, signer, previous):
        self.transactions = transactions
        self.signer = signer
        self.previous = previous
        
    def to_dict(self):
        return {'transactions':list(x.__dict__ for x in self.transactions), 'signer':self.signer, 'previous':self.previous}
    
    def __str__(self):
        return str(self.to_dict())