from abc import ABC, abstractmethod


class AlphaBeta(ABC):

    @abstractmethod
    def heuristicFunction(self, nodeToEvaluate):
        pass

    @abstractmethod
    def nextNodesGenerator(self, currentNode):
        pass

    @abstractmethod
    def nextTurn(self, currentNode):  # pentru a da randul oponentului, returneaza None daca jocul este la final
        pass

    @abstractmethod
    def finished(self):
        pass

    def __init__(self, startNode, DFH):  # nextNodesGenerator returneaza +inf daca e stare finala

        self.start = startNode  # starea de start
        self.nextOptions = {}  # pentru fiecare runda a lui max, pentru a alege cea mai buna urmatoare miscare

        self.DFH = DFH  # depth first height

    def df(self, currentNode, currentDepth, alpha, beta, player, saveState):

        estimatedValue = self.heuristicFunction(currentNode)
        if currentDepth == 0 or estimatedValue == float('inf') or estimatedValue == float('-inf'):
            return estimatedValue

        if player == 'MAX':

            estimatedValue = float('-inf')

            for nextNode in self.nextNodesGenerator(currentNode):

                estimatedValue = max(estimatedValue, self.df(nextNode, currentDepth - 1, alpha, beta, 'MIN', max(0, saveState - 1)))
                alpha = max(alpha, estimatedValue)

                if alpha >= beta:
                    break

        elif player == 'MIN':

            estimatedValue = float('inf')

            for nextNode in self.nextNodesGenerator(currentNode):

                estimatedValue = min(estimatedValue, self.df(nextNode, currentDepth - 1, alpha, beta, 'MAX', max(0, saveState - 1)))
                beta = min(beta, estimatedValue)

                if alpha >= beta:
                    break

        if saveState == 1 and estimatedValue not in self.nextOptions.keys():
            self.nextOptions.update({estimatedValue: currentNode})

        return estimatedValue

    def go(self):

        currentNode = self.start
        while True:

            self.nextOptions = {}

            bestNextValue = self.df(currentNode, self.DFH, float('-inf'), float('inf'), 'MAX', 2)
            nextMove = self.nextOptions[bestNextValue]

            opponentChoice = self.nextTurn(nextMove)
            if opponentChoice is None:
                break

            currentNode = opponentChoice








