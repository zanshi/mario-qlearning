package tnm095;

import ch.idsia.agents.Agent;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.benchmark.mario.engine.sprites.Mario;

/**
 * Created by Niclas Olmenius on 2017-09-25.
 */
public class QLearningAgent extends BasicMarioAIAgent implements Agent {


    public QLearningAgent() {
        super("Q-Learning Agent");
        reset();
    }

    @Override
    public boolean[] getAction() {
        return super.getAction();
    }

    @Override
    public void reset() {
        super.reset();
    }

}
