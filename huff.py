class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq  
        self.symbol = symbol  
        self.left = left  
        self.right = right  
        self.huff = ''  

def printNodes(node, val=''):
    newVal = val + str(node.huff)
    if node.left:
        printNodes(node.left, newVal)
    if node.right:
        printNodes(node.right, newVal)
    if not node.left and not node.right:  
        print(f"{node.symbol} -> {newVal}")

# Characters for Huffman Tree
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# Frequency of characters
freq = [4, 7, 12, 14, 17, 43, 54]

nodes = []


for x in range(len(chars)):
    nodes.append(Node(freq[x], chars[x]))

# Build the Huffman Tree
while len(nodes) > 1:
    
    nodes = sorted(nodes, key=lambda x: x.freq)
 
    left = nodes[0]
    right = nodes[1]
  
    left.huff = 0
    right.huff = 1
   
    newNode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
    
    nodes.remove(left)
    nodes.remove(right)
    nodes.append(newNode)


print("Huffman Codes:")
printNodes(nodes[0])
