from uuid import uuid4
from flask import Flask, jsonify, request

from consensus import Consensus
from transaction import Transaction

import pdb

# Instantiate the Node
web_app = Flask(__name__, static_url_path = "")
my_app = Consensus()

# Generate a globally unique address for this node
node_identifier = str(uuid4()).replace('-', '')

# transaction/new
@web_app.route('/transaction/new', methods=['POST'])
def new_transaction():
    values = request.get_json()

    # Check that the required fields are in the POST'ed data
    required = ['sender', 'recipient', 'code']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # Create a new Transaction
    transaction = Transaction(values['sender'], values['recipient'], values['code'])
    approved = my_app.new_transaction(transaction)
    if not approved:
        return 'unauthorized transaction', 400

    response = {'message': 'Transaction added successfully'}
    return jsonify(response), 201

def print_dict(my_dict):
    return '\n\t'.join(str(key)+' : '+(str(my_dict[key]) if my_dict[key]<2**16 else bytes.fromhex(hex(my_dict[key])[2:]).decode()) for key in my_dict)
    
# transaction/get
@web_app.route('/transaction/get', methods=['GET'])
def get_transaction():
    values = request.get_json()

    # Check that the required fields are in the POST'ed data
#    required = ['index']
#    if not all(k in values for k in required):
#        return 'Missing values', 400

#    response = '\n'.join(str(o) for o in my_app.blockchain.get_all())
#    response = '\ncode-\n'+'\n'.join(key.hex()[-5:]+' : '+my_app.state.code[key].hex()[1:40] for key in my_app.state.code)+'\n\nstorage-\n'+\
#               '\n'.join(key.hex()[-5:]+' : \n\t'+print_dict(my_app.state.storage[key]) for key in my_app.state.storage)
    response = {'message': 'Transaction added successfully'}
    return jsonify(response), 200


# network/report
@web_app.route('/network/report', methods=['POST'])
def network_report():
    values = request.get_json()

    # Check that the required fields are in the POST'ed data
    required = ['sender', 'recipient', 'code', 'registrer', 'notifier']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # Create a new Transaction
    transaction = Transaction(values['sender'], values['recipient'], values['code'], values['registrer'], values['notifier'])
    # check with app
    my_app.report_transaction(transaction)

    response = {'message': 'cypy that transaction'}
    return jsonify(response), 201

    
# network/sign
@web_app.route('/network/sign', methods=['POST'])
def network_sign():
    values = request.get_json()

    # Check that the required fields are in the POST'ed data
    required = ['transactions', 'signer', 'previous']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # check with app
    transaction_list = values['transactions']
    transactions = []
    for transaction in transaction_list:
        transactions.append(Transaction(transaction['sender'], transaction['recipient'], transaction['code'], transaction['registrer'], transaction['notifier']))

    my_app.report_block(transactions, values['signer'], values['previous'])

    response = {'message': 'copy that block'}
    return jsonify(response), 201

    
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    my_app.set_port(port)
    web_app.run(host='0.0.0.0', port=port)
