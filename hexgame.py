import copy
from collections import deque


class Board:

    @staticmethod
    def fastHeuristic(self):

        final, winner = self.isFinalState()

        if final is True and winner == self.LRP:
            return float('inf')
        elif final is True and winner == self.UDP:
            return float('-inf')

        # calculez distanta absoluta pentru fieicare portiune continua de B sau R

        minUpUDP = 1 + self.size ** 2
        minDownUDP = 1 + self.size ** 2

        minLeftLRP = 1 + self.size ** 2
        minRightLRP = 1 + self.size ** 2

        for i in range(self.size):
            for j in range(self.size):

                if self.board[i][j] == self.LRP:
                    if j < minLeftLRP:
                        minLeftLRP = j
                    if self.size - j - 1 < minRightLRP:
                        minRightLRP = self.size - j - 1

                if self.board[i][j] == self.UDP:
                    if i < minUpUDP:
                        minUpUDP = i
                    if self.size - i - 1 < minDownUDP:
                        minDownUDP = self.size - i - 1

        return 1.3 ** (minRightLRP + minLeftLRP) - 1.6 ** (100000 / (minDownUDP + minDownUDP))

    @staticmethod
    def heuristic(self):

        final, winner = self.isFinalState()

        if final is True and winner == self.LRP:
            return float('inf')
        elif final is True and winner == self.UDP:
            return float('-inf')

        # pentru fiecare bucata continua de B, voi calcula distanta minima pana formez un lant castigator
        # fac minimul intre aceste minime

        u = [-1, -1, 0, 0, 1, 1]
        v = [0, 1, -1, 1, -1, 0]

        matrix = copy.deepcopy(self.board)
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]

        def valid(x, y):
            return 0 <= x < self.size and 0 <= y < self.size and visited[x][y] is False

        def fill(x, y):

            visited[x][y] = True

            for step in range(6):
                if valid(x + u[step], y + v[step]) is True and matrix[x + u[step]][y + v[step]] == self.LRP:
                    fill(x + u[step], y + v[step])

        def getMinDistance(x, y):

            m = [[None for _ in range(self.size)] for _ in range(self.size)]
            auxvisited = [[False for _ in range(self.size)] for _ in range(self.size)]

            for i in range(self.size):
                for j in range(self.size):

                    if matrix[i][j] == self.LRP:
                        m[i][j] = 0
                    elif matrix[i][j] == self.UDP:
                        m[i][j] = float('inf')

            minLeft = 1 + self.size ** 2   # distanta minima pana la latura stanga
            minRight = 1 + self.size ** 2  # distanta minima pana la latura dreapta

            q = deque()

            q.append((x, y))
            auxvisited[x][y] = True

            while q:

                current = q.popleft()

                if current[1] == 0:

                    minLeft = min(minLeft, m[current[0]][current[1]])
                    continue

                if current[1] == self.size - 1:

                    minRight = min(minRight, m[current[0]][current[1]])
                    continue

                for step in range(6):

                    inext = current[0] + u[step]
                    jnext = current[1] + v[step]

                    if 0 <= inext < self.size and 0 <= jnext < self.size:

                        if m[inext][jnext] == 0 and auxvisited[inext][jnext] is False:

                            q.appendleft((inext, jnext))
                            auxvisited[inext][jnext] = True

                        elif m[inext][jnext] is None or m[inext][jnext] > m[current[0]][current[1]] + 1:

                            q.append((inext, jnext))
                            m[inext][jnext] = m[current[0]][current[1]] + 1
                            auxvisited[inext][jnext] = True

            return minLeft + minRight

        minDistance = 1 + self.size ** 2

        for i in range(self.size):
            for j in range(self.size):

                if matrix[i][j] == self.LRP:

                    minDistance = min(minDistance, getMinDistance(i, j))
                    fill(i, j)

        return minDistance

    '''def heuristic2(self):

        final, winner = self.isFinalState()

        if final is True and winner == self.LRP:
            return float('inf')
        elif final is True and winner == self.UDP:
            return float('-inf')

        # pentru fiecare bucata continua de B, voi calcula distanta minima pana formez un lant castigator
        # fac minimul intre aceste minime

        u = [-1, -1, 0, 0, 1, 1]
        v = [0, 1, -1, 1, -1, 0]

        matrix = copy.deepcopy(self.board)
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]

        def valid(x, y):
            return 0 <= x < self.size and 0 <= y < self.size and visited[x][y] is False

        def fill(x, y):

            visited[x][y] = True

            for step in range(6):
                if valid(x + u[step], y + v[step]) is True and matrix[x + u[step]][y + v[step]] == self.LRP:
                    fill(x + u[step], y + v[step])

        def getMinDistance(x, y):

            m = [[None for _ in range(self.size)] for _ in range(self.size)]
            auxvisited = [[False for _ in range(self.size)] for _ in range(self.size)]

            for i in range(self.size):
                for j in range(self.size):

                    if matrix[i][j] == self.LRP:
                        m[i][j] = 0
                    elif matrix[i][j] == self.UDP:
                        m[i][j] = float('inf')

            minLeft = 1 + self.size ** 2   # distanta minima pana la latura stanga
            minRight = 1 + self.size ** 2  # distanta minima pana la latura dreapta

            q = deque()

            q.append((x, y))
            auxvisited[x][y] = True

            while q:

                current = q.popleft()

                if current[1] == 0:

                    minLeft = min(minLeft, m[current[0]][current[1]])
                    continue

                if current[1] == self.size - 1:

                    minRight = min(minRight, m[current[0]][current[1]])
                    continue

                for step in range(6):

                    inext = current[0] + u[step]
                    jnext = current[1] + v[step]

                    if 0 <= inext < self.size and 0 <= jnext < self.size:

                        if m[inext][jnext] == 0 and auxvisited[inext][jnext] is False:

                            q.appendleft((inext, jnext))
                            auxvisited[inext][jnext] = True

                        elif m[inext][jnext] is None or m[inext][jnext] > m[current[0]][current[1]] + 1:

                            q.append((inext, jnext))
                            m[inext][jnext] = m[current[0]][current[1]] + 1
                            auxvisited[inext][jnext] = True

            return minLeft + minRight

        minDistance = 1 + self.size ** 2

        for i in range(self.size):
            for j in range(self.size):

                if matrix[i][j] == self.LRP:

                    minDistance = min(minDistance, getMinDistance(i, j))
                    fill(i, j)

        return minDistance '''

    def nextStatesGenerator(self, currentPlayer):

        nextStates = []

        currentColor = self.LRP
        if currentPlayer == 'MIN':
            currentColor = self.UDP

        for i in range(self.size):
            for j in range(self.size):

                if self.board[i][j] is None:

                    nextMatrix = copy.deepcopy(self.board)
                    nextMatrix[i][j] = currentColor

                    nextState = Board(nextMatrix, self.size)
                    nextStates.append(nextState)

        return nextStates

    def isFinalState(self):

        visited = [[False for _ in range(self.size)] for _ in range(self.size)]

        def checkWin(x, y, player):

            u = [-1, -1, 0, 0, 1, 1]
            v = [0, 1, -1, 1, -1, 0]

            def valid(x, y):
                return 0 <= x < self.size and 0 <= y < self.size and visited[x][y] is False

            visited[x][y] = True

            if player == self.LRP and y == self.size - 1:
                return True
            elif player == self.UDP and x == self.size - 1:
                return True

            won = False
            for step in range(6):
                if valid(x + u[step], y + v[step]) and self.board[x + u[step]][y + v[step]] == player:
                    won = won or checkWin(x + u[step], y + v[step], player)

            return won

        for i in range(self.size):
            if self.board[i][0] == self.LRP:
                won = checkWin(i, 0, self.LRP)
                if won:
                    return True, self.LRP

        for j in range(self.size):
            if self.board[0][j] == self.UDP:
                won = checkWin(0, j, self.UDP)
                if won:
                    return True, self.UDP

        return False, None

    def __init__(self, matrix=None, size=11):

        if matrix is None:
            self.board = [[None for _ in range(size)] for _ in range(size)]
        else:
            self.board = matrix

        self.size = size

        self.LRP = 'B'
        self.UDP = 'R'


# real player - R (up-down)
# AI          - B (left-right)
class Hexgame:

    def nextTurn(self, currentNode):  # pentru a da randul oponentului, returneaza None daca jocul este la final

        print("your turn:")
        for i in range(currentNode.size):

            print(' ' * i, end='')

            for j in range(currentNode.size):

                toPrint = '-'
                if currentNode.board[i][j] is not None:
                    toPrint = currentNode.board[i][j]

                print(toPrint + ' ', end='')

            print()

        print("choose next move: ")

        while True:

            x, y = (int(el) for el in input().split())
            if 0 <= x < currentNode.size and 0 <= y < currentNode.size and currentNode.board[x][y] is None:

                newBoard = copy.deepcopy(currentNode.board)
                newBoard[x][y] = self.UDP

                return Board(newBoard, currentNode.size)

            else:
                print("invalid move; choose another action")

    @staticmethod
    def finish(winner):

        print(f"game won by {winner}")
        quit()

    def __init__(self, fstPlayer, dimension=11, DFH=5):

        self.fstPlayer = fstPlayer

        self.start = Board(size=dimension)
        self.DFH = DFH

        self.nextOptions = {}

        self.UDP = 'R'
        self.LRP = 'B'

        self.h = Board.fastHeuristic

    @staticmethod
    def count(currentState):

        cnt = 0

        for i in range(currentState.size):
            for j in range(currentState.size):
                if currentState.board[i][j] != "-":
                    cnt += 1

        return cnt

    def df(self, currentBoard, currentDepth, alpha, beta, player, saveState):

        estimatedValue = self.h(currentBoard)
        if currentDepth == 0 or estimatedValue == float('inf') or estimatedValue == float('-inf'):
            return estimatedValue

        nextBoards = []

        if player == 'MAX':

            nextBoards = currentBoard.nextStatesGenerator('MAX')
            for i in range(len(nextBoards)):
                nextBoards[i] = [0, nextBoards[i]]  # [val, node]

            estimatedValue = float('-inf')
            for nextNode in nextBoards:

                estimatedValue = max(estimatedValue, self.df(nextNode[1], currentDepth - 1, alpha, beta, 'MIN', False))
                alpha = max(alpha, estimatedValue)

                nextNode[0] = estimatedValue

                if alpha >= beta:
                    break

        elif player == 'MIN':

            nextBoards = currentBoard.nextStatesGenerator('MIN')
            for i in range(len(nextBoards)):
                nextBoards[i] = [0, nextBoards[i]]  # [node] = [val=0, node]

            estimatedValue = float('inf')
            for nextNode in nextBoards:

                estimatedValue = min(estimatedValue, self.df(nextNode[1], currentDepth - 1, alpha, beta, 'MAX', False))
                beta = min(beta, estimatedValue)

                nextNode[0] = estimatedValue

                if alpha >= beta:
                    break

        if saveState is True:
            for next in nextBoards:
                self.nextOptions.update({next[0]: next[1]})

        return estimatedValue

    def go(self):

        if self.fstPlayer == self.UDP:
            currentNode = self.nextTurn(self.start)
        else:
            currentNode = self.start

        while True:

            self.nextOptions = {}

            if self.count(currentNode) > (currentNode.size ** 2) // 2:
                self.h = Board.heuristic

            bestNextValue = self.df(currentNode, self.DFH, float('-inf'), float('inf'), 'MAX', True)
            nextMove = self.nextOptions[bestNextValue]

            t = nextMove.isFinalState()
            if t[0] is True:
                Hexgame.finish(t[1])

            opponentChoice = self.nextTurn(nextMove)

            t = opponentChoice.isFinalState()
            if t[0] is True:
                Hexgame.finish(t[1])

            currentNode = opponentChoice


game = Hexgame(fstPlayer='B', dimension=5, DFH=5)
game.go()





