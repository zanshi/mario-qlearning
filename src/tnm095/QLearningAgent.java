package tnm095;

import ch.idsia.agents.Agent;
import ch.idsia.agents.LearningAgent;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.benchmark.tasks.LearningTask;

import java.util.Hashtable;

/**
 * Created by Niclas Olmenius on 2017-09-25.
 */
public class QLearningAgent extends BasicMarioAIAgent implements LearningAgent {

    private class QState {
        QState(int[] s) {
            states = s;
        }

        int[] states;

    }

    private class QAction {
        boolean[] actions;
    }

    private class QStateAction {
        QStateAction(QState q, QAction a) {
            qState = q;
            qAction = a;
        }

        QState qState;
        QAction qAction;

    }

    float[] previousMarioPos;

    Hashtable<QStateAction, Float> qTable;


    public QLearningAgent() {
        super("Q-Learning Agent");
        qTable = new Hashtable<>();
        reset();
    }

    private float getReward(QState state) {

        float d =


    }

    @Override
    public boolean[] getAction() {
        return super.getAction();
    }

    @Override
    public void reset() {
        super.reset();
    }

    @Override
    public void learn() {

        //


    }

    @Override
    public void giveReward(float reward) {

    }

    @Override
    public void newEpisode() {

    }

    @Override
    public void setLearningTask(LearningTask learningTask) {

    }

    @Override
    public void setEvaluationQuota(long num) {

    }

    @Override
    public Agent getBestAgent() {
        return null;
    }

    @Override
    public void init() {

    }
}
