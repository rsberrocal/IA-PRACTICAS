# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # Get the successor from the actions
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # Get position of pacman and foods
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        # Get the ghosts states
        newGhostStates = successorGameState.getGhostStates()
        # Timer of every ghost that we can eat
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Get the rest of the lists
        foods = newFood.asList()

        # Calculate the distance for each food
        food_dist = [manhattanDistance(food, newPos) for food in foods]
        # Get the distance for each ghost, only if we can eat it
        ghost_dist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if
                      newScaredTimes[0] == 0]
        # Get all the current food
        current_foods = currentGameState.getFood().asList()
        # Get current capsules
        current_caps = currentGameState.getCapsules()
        # Get capsules after the actions
        new_caps = successorGameState.getCapsules()
        # Aux variables
        score = 0
        min_food = float('inf')

        # STATES
        # If we eat food, we plus the score
        if len(foods) < len(current_foods):
            score += 100

        # If we eat a capsule, we plus the score
        if len(new_caps) < len(current_caps):
            score += 1000

        # If we win we return the maximum score
        if successorGameState.isWin():
            return 10000

        # If we stop we reduce the score
        if action == Directions.STOP:
            score -= 100

        # Check the distance
        # If the ghost is near to pacman we reduce the score
        for ghost in ghost_dist:
            if ghost < 4:
                score -= 10000

        # Finally check foods
        for food in food_dist:
            if food < min_food:
                min_food = food

        # The result is the score plus 100 to move and reduce the closest food.
        # If the food is far this action will be lower
        return score + 100 - min_food


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Get the number of ghost
        num_ghost = gameState.getNumAgents() - 1

        def maxAgent(gameState, depth):
            # In case we are winning or not we return the evaluationFunction
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Aux variables
            best_action = None
            best_score = float('-inf')

            # Get the legal actions for pacman
            legal_action = gameState.getLegalActions(0)  # 0 is the index for pacman

            # Iterate each action
            for action in legal_action:
                # Get the succesor
                succ = gameState.generateSuccessor(0, action)
                # Get the value for the action
                v = minAgent(succ, depth, 1)

                # Only we take the max value
                if v > best_score:
                    best_score = v
                    best_action = action

            # If we are on Top return the action
            # Else, return the score
            if depth == 0:
                return best_action
            else:
                return best_score

        def minAgent(gameState, depth, ghost):
            # In case we are winning or not we return the evaluationFunction
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Aux variables
            best_score = float('inf')

            # Get the legal actions for ghosts
            # The ghosts are from 1 - num of agents - 1
            legal_actions = gameState.getLegalActions(ghost)

            # Iterate each action
            for action in legal_actions:
                # Get the successor
                succ = gameState.generateSuccessor(ghost, action)

                # If we still have ghost, get the next value for the successor
                if ghost < num_ghost:
                    v = minAgent(succ, depth, ghost + 1)
                else:
                    # If we are on bottom of the depth the value will be the evaluation function
                    if depth == self.depth - 1:
                        v = self.evaluationFunction(succ)
                    else:
                        # If not, turn to pacman
                        v = maxAgent(succ, depth + 1)
                # Calculate the best score in each action
                best_score = min(v, best_score)
            return best_score

        return maxAgent(gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Get the number of ghost
        num_ghost = gameState.getNumAgents() - 1

        alpha = float('-inf')
        beta = float('inf')

        def maxAgent(gameState, depth, alpha, beta):
            # In case we are winning or not we return the evaluationFunction
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Aux variables
            best_action = None
            best_score = float("-inf")

            # Get the legal actions for pacman
            legal_actions = gameState.getLegalActions(0)

            # Iterate each action
            for action in legal_actions:
                # Get the succesor
                succ = gameState.generateSuccessor(0, action)
                # Get the value for the action
                v = minAgent(succ, depth, 1, alpha, beta)
                # Only we take the max value
                if v > best_score:
                    best_score = v
                    best_action = action

                # Beta condition
                # If the score is more than beta we cut
                if best_score > beta:
                    return best_score
                # Change alpha
                alpha = max(alpha, best_score)

            # If we are on Top return the action
            # Else, return the score
            if depth == 0:
                return best_action
            else:
                return best_score

        def minAgent(gameState, depth, ghost, alpha, beta):
            # In case we are winning or not we return the evaluationFunction
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Aux variables
            best_score = float("inf")

            # Get the legal actions for ghosts
            # The ghosts are from 1 - num of agents - 1
            legal_actions = gameState.getLegalActions(ghost)

            # Iterate each action
            for action in legal_actions:
                # Get the successor
                succ = gameState.generateSuccessor(ghost, action)

                # If we still have ghost, get the next value for the successor
                if ghost < num_ghost:
                    v = minAgent(succ, depth, ghost + 1, alpha, beta)
                else:
                    # If we are on bottom of the depth the value will be the evaluation function
                    if depth == self.depth - 1:
                        v = self.evaluationFunction(succ)
                    else:
                        # If not, turn to pacman
                        v = maxAgent(succ, depth + 1, alpha, beta)

                # Calculate the best score in each action
                best_score = min(v, best_score)

                # If best score is lower than alpha cut
                if best_score < alpha:
                    return best_score
                beta = min(beta, best_score)

            return best_score

        return maxAgent(gameState, 0, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, action, depth, agent_index):
            # In case we are winning or not we return the evaluationFunction or we are on depth 0
            if depth is 0 or gameState.isWin() or gameState.isLose():
                return action, self.evaluationFunction(gameState)

            # See which agent we are playing
            # Pacman is 0
            if agent_index is 0:
                return maxvalue(gameState, action, depth, agent_index)
            else:
                return expvalue(gameState, action, depth, agent_index)

        def maxvalue(gameState, action, depth, agent_index):
            # Possible actions will be now a tuple with two elements
            # [0] action, [1] cost
            best_action = ("max", float("-inf"))

            # Iterate each action
            for legal_action in gameState.getLegalActions(agent_index):
                # Get next agent
                next_agent = (agent_index + 1) % gameState.getNumAgents()
                # If we are on bot of depth take the action
                if depth != self.depth * gameState.getNumAgents():
                    succ = action
                else:
                    # Else, take the possible action
                    succ = legal_action
                # Get value
                v = expectimax(gameState.generateSuccessor(agent_index, legal_action), succ, depth - 1, next_agent)
                # Get action
                best_action = max(best_action, v, key=lambda x: x[1])

            return best_action

        def expvalue(gameState, action, depth, agent_index):
            # Get legal actions for the agent
            legal_actions = gameState.getLegalActions(agent_index)
            # Variables to calculate probabilities
            avr = 0
            prob = 1.0 / len(legal_actions)

            # Iterate each action
            for legal in legal_actions:
                # Get next agent
                next_agent = (agent_index + 1) % gameState.getNumAgents()
                # Get action
                best_action = expectimax(gameState.generateSuccessor(agent_index, legal), action, depth - 1,
                                         next_agent)

                # Calculate the probability of an action
                avr += best_action[1] * prob
            return action, avr

        max_depth = self.depth * gameState.getNumAgents()

        return expectimax(gameState, "expect", max_depth, 0)[0]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The final score is divided in two parts
    First when we can eat the ghost and then normal ghosts
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Manhattan distance to the foods from the current state
    # Get the rest of the lists
    food_list = newFood.asList()
    # Calculate the distance of each food
    food_distance = [manhattanDistance(newPos, pos) for pos in food_list]

    # Manhattan distance to each ghost from the current state
    ghost_pos = [ghost.getPosition() for ghost in newGhostStates]
    # Calculate the distance of each ghost
    ghost_distance = [manhattanDistance(newPos, pos) for pos in ghost_pos]

    # Get the num of capsules
    num_capsules = len(currentGameState.getCapsules())

    # Score
    score = 0
    # Get number of food that we didnt eat
    num_food_no_eaten = len(newFood.asList(False))
    # Sum all the times of the scared ghost
    sum_scare_times = sum(newScaredTimes)
    # Sum all the distance of each ghost
    sum_ghost_dist = sum(ghost_distance)

    # Sum all the score that we have with the number of food we didnt eat
    score += currentGameState.getScore() + num_food_no_eaten

    # if we have ghost that we can eat
    if sum_scare_times > 0:
        # We give more importance to these ghosts
        score += sum_scare_times + (-1 * num_capsules) + (-1 * sum_ghost_dist)
    else:
        score += sum_ghost_dist + num_capsules
    return score


# Abbreviation
better = betterEvaluationFunction
