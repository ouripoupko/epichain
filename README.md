# epichain
In order to answer the question - what are the minimal components of an on chain governed blockchain, I developed a thin blockchain implementation, which I called 'epichain'.

 

The blockchain has the following main components:

 

A network layer that receives transactions through REST

A local copy of the evm (Ethereum virtual machine)

Internal state implementation

A minimal block run time storage

A naive consensus protocol

A demonstrative client

 

To demonstrate on chain governance, the internal state includes a deployment of four smart contracts, which are stored on the genesis block:

 

Params - a smart contract that stores key-value pairs, initialized with the addresses of the following three contracts.

Permission - check if a request to write a parameter to the 'params' smart contract has the permission to do so

CheckTx and DeliverTx - intended to act as an implementation of a two-phase commit, like in pBFT.

 

The naive consensus protocol, which assumes no threats at this time, simply lets each miner in the list of miners to act as a block signer in its turn. The protocol calls the smart contracts in the genesis block to validate incoming transactions. Note that since the address of these contracts is written in the ledger (in the runtime state of the 'params' contract), the users of the blockchain are able to deploy new contracts that will change the behavior of the consensus protocol by overriding the contracts in the genesis block.

 

When initializing the blockchain it is completely permissive - anyone can write to the 'params' contract and all transactions are accepted. The demonstrative client deploys the following contracts to the blockchain – persona, community and voting. It then overrides the 'permission' contract with a new contract that accepts write requests only if majority of the personas in the community signed them. From that point in time the blockchain becomes restrictive.

 

Notice that implementing a consensus protocol within the ledger is very limited in this approach. The virtual machine is by definition closed, like a sandbox environment – it has no access to events outside the memory of the computer. In particular, it has no network access to the other nodes in the network. Otherwise, the state of the VM cannot be deterministic.
