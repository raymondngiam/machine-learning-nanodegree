import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QValueDict(dict):
    """Modified dictionary class to store Q values ."""
    def __getitem__(self, idx):
        """ 
          Handle exception of dictionary invalid key error,
          if invalid key is used, return 0. 		  
        """
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def argMax(self):
        """ 
          Return the key of the highest value	  
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = QValueDict()
        self.alpha=0.1
        self.discount=0.1
        self.epsilon=0.1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def getQValue(self,state,action):
        """ Returns Q value for particular (state, action) pair from QValueDict."""
        return self.Q[state,action]

    def computeValueFromQValues(self, state):
        """ Returns max Q value for a particular state, where max is over all possible actions."""
        qValues = QValueDict()
        actions =  [None,'forward','left','right']
        for action in actions:
            qValues[action] = self.getQValue(state,action)
        maxAction = qValues.argMax()
        return qValues[maxAction]

    def computeActionFromQValues(self, state):
        """ Returns the best action (i.e. the one with highest Q value) to be taken in a state."""
        qValues = QValueDict()
        actions =  [None,'forward','left','right']
        for action in actions:
            qValues[action] = self.getQValue(state,action)
        maxAction = qValues.argMax()
        return maxAction

    def flipCoin(self, p):
        """ 
          Returns boolean of binomial distribution with success probability, p. 
          To be used for determining trade off between exploration and exploitation in Q learning.
        """
        r = random.random()
        return r < p

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'],inputs['left'],self.next_waypoint)
        
        # TODO: Select action according to your policy
        actions =  [None,'forward','left','right']
        bolExplore = self.flipCoin(self.epsilon)
        if (bolExplore):
            action = random.choice(actions)
        else:
            action = self.computeActionFromQValues(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Gather next waypoint and environment inputs after action is taken
        nextState_next_waypoint = self.planner.next_waypoint()
        nextState_inputs = self.env.sense(self)
        # Store as next state
        nextState = (nextState_inputs['light'], nextState_inputs['oncoming'],nextState_inputs['left'],nextState_next_waypoint)
        # Update Q values 
        self.Q[self.state,action] = (1-self.alpha)*self.Q[self.state,action] + self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

def finetune():
    """Fine tune the agent with parameter grid search."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    n = 100
    alphas = [0.1,0.5,0.9]
    discounts = [0.1,0.5,0.9]
    epsilons = [0.1]
    result = {}
    for i in range(len(alphas)):
        a.alpha = alphas[i]
        for j in range(len(discounts)):
            a.discount = discounts[j]
            for k in range(len(epsilons)):
                a.epsilon = epsilons[k]
                sim.run(n_trials=n)  # run for a specified number of trials
                result["{},{},{}".format(a.alpha,a.discount,a.epsilon)] = float(e.success)/n
                e.success=0  # reset success counter in environment object
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "Final success rates: {}".format(result)

if __name__ == '__main__':
    #run()
    finetune()
