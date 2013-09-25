# Bradley N. Miller, David L. Ranum
# Introduction to Data Structures and Algorithms in Python
# Copyright 2005
# 


__all__=['BinaryTree', 'Stack']


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)



class BinaryTree:
    """
    A recursive implementation of Binary Tree
    Using links and Nodes approach.
    """    
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
        self.parent = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode) if not isinstance(newNode, BinaryTree) else newNode
        else:
            t = BinaryTree(newNode) if not isinstance(newNode, BinaryTree) else newNode
            # t.left = self.leftChild
            self.leftChild = t
        self.leftChild.parent = self    
    
    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode) if not isinstance(newNode, BinaryTree) else newNode
        else:
            t = BinaryTree(newNode) if not isinstance(newNode, BinaryTree) else newNode
            # t.right = self.rightChild
            self.rightChild = t
        self.rightChild.parent = self    

    def isLeaf(self):
        return ((not self.leftChild) and (not self.rightChild))

    def isRoot(self):
        return not self.parent
        
    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild
        
    def getSibling(self):
        if self.isRoot(): 
            return None
        Rsib = self.parent.getRightChild() 
        return Rsib if Rsib != self else self.parent.getLeftChild()

    def hasChild(self):
        return (self.rightChild != None) or (self.leftChild != None) 

    def hasParent(self):
        return (self.key != None)

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

    def inorder(self):
        if self.leftChild:
            self.leftChild.inorder()
        print(self.key)
        if self.rightChild:
            self.rightChild.inorder()

    def postorder(self):
        if self.leftChild:
            self.leftChild.postorder()
        if self.rightChild:
            self.rightChild.postorder()
        print(self.key)


    def preorder(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()

    def printexp(self):
        sVal = ""
        if self:
            sVal = '('  if self.hasChild() else ''
            sVal += printexp(self.getLeftChild())
            sVal = sVal + str(self.getRootVal())
            sVal = sVal + printexp(self.getRightChild()) 
            sVal += ')' if self.hasChild() else ''
        return sVal

    def __str__(self):
        return self.printexp()


    def postordereval(self, opers = None):
        if not opers:
            opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
        res1 = None
        res2 = None
        if self.leftChild:
            res1 = self.leftChild.postordereval()  #// \label{peleft}
        if self.rightChild:
            res2 = self.rightChild.postordereval() #// \label{peright}
        if res1 and res2:
            return opers[self.key](res1,res2) #// \label{peeval}
        else:
            return self.key

def inorder(tree):
    if tree != None:
        inorder(tree.getLeftChild())
        print(tree.getRootVal())
        inorder(tree.getRightChild())

# def printexp(tree):
#     if tree.leftChild:
#         print'( '
#         printexp(tree.getLeftChild())
#     print '%s '%tree.getRootVal()
#     if tree.rightChild:
#         printexp(tree.getRightChild())
#         print') '

def printexp(tree):
    sVal = ""
    if tree:
        sVal = '('  if tree.hasChild() else ''
        sVal += printexp(tree.getLeftChild())
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + printexp(tree.getRightChild()) 
        sVal += ')' if tree.hasChild() else ''
    return sVal

def postordereval(tree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    res1 = None
    res2 = None
    if tree:
        res1 = postordereval(tree.getLeftChild())  #// \label{peleft}
        res2 = postordereval(tree.getRightChild()) #// \label{peright}
        if res1 and res2:
            return opers[tree.getRootVal()](res1,res2) #// \label{peeval}
        else:
            return tree.getRootVal()

def height(tree):
    if tree == None:
        return -1
    else:
        return 1 + max(height(tree.leftChild),height(tree.rightChild))

if __name__ == '__main__':
    t = BinaryTree(7)
    t.insertLeft(3)
    t.insertRight(9)
    inorder(t)
    # import operator
    x = BinaryTree('*')
    x.insertLeft('+')
    l = x.getLeftChild()
    l.insertLeft(4)
    l.insertRight(5)
    x.insertRight(7)
    print(printexp(x))
    # print(postordereval(x))
    print(height(x))
