

class Transaction:
    def __init__(self, sender, recipient, code, registrer = None, notifier = None):
        self.sender = sender
        self.recipient = recipient
        self.code = code
        self.registrer = registrer
        self.notifier = notifier

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other): 
        return [self.sender,self.recipient,self.code] == [other.sender,other.recipient,other.code]

    def set_registrer(self, registrer):
        self.registrer = registrer
    
    def set_notifier(self, notifier):
        self.notifier = notifier