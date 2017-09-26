package tnm095;

import ch.idsia.agents.Agent;
import ch.idsia.agents.LearningAgent;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.benchmark.tasks.LearningTask;

import java.util.Arrays;
import java.util.Hashtable;
import java.util.Random;

/**
 * Created by Niclas Olmenius on 2017-09-25.
 */
public class QLearningAgent extends BasicMarioAIAgent implements LearningAgent {

    // Relevant data structures
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

    // -----------------------------------------------
    // Member variables

    // Mario AI
    private float[] prevPos;
    private boolean killed;
    private boolean killedWithFire;
    private boolean killedWithStomp;
    private boolean killedWithShell;

    LearningTask task;
    long evalQuota;

    // Q-learning
    public static final Random random = new Random();
    private Hashtable<QStateAction, Float> QTable;

    private QState prevS;
    private QAction prevA;
    private float prevR;

    private float alpha;
    private float gamma;
    private float rho;

    QState currState;

    // -----------------------------------------------
    // Ctor
    public QLearningAgent() {
        super("Q-Learning Agent");
        QTable = new Hashtable<>();
        alpha = 0.3f;
        gamma = 0.75f;
        rho = 0.3f;

        killed = false;
        killedWithFire = false;
        killedWithStomp = false;
        killedWithShell = false;

        reset();
    }

    // TODO: Tweak weights
    // Reward function. Weighted value based on the input state.
    private float getReward(QState state) {
        int[] states = state.states;
        int nStates = states.length;

        // Define weights
        float[] weights = new float[nStates];
        Array.fill(weights,1.f/nStates);

        // Calculate delta distance
        float dx = marioFloatPos[0] - prevPos[0];
        float dy = marioFloatPos[1] - prevPos[1];

        float movementReward;
        if (getEnemiesCellValue(marioEgoRow, marioEgoCol) != 0) { // colliding with enemy
            movementReward = -1.f;
        } else if (dx*dx + dy*dy < 0.00001f) { // being stuck
            movementReward = -0.5;
        } else if (dx > 0.f && (getEnemiesCellValue(marioEgoRow, marioEgoCol+1) != 0 || 
        getEnemiesCellValue(marioEgoRow, marioEgoCol+2) != 0)) { // moving forward when enemy is present
            movementReward = -0.25f;
        } else {
            movementReward = dx + dy;
        }

        // Calculate reward
        float reward = movementReward * 1.f; // some arbitrary weight
        for(int i = 0; i < nStates; i++) {
            reward += weights[i] * states[i];
        }
        return reward;
    }

    private int calculateDirection() {

        // Counter clockwise
        // 0: Standing still
        // 1: South west
        // 2: West
        // 3: North west
        // 4: North
        // 5: North east
        // 6: East
        // 7: South east
        // 8: South

        if (prevPos == null || (Arrays.equals(prevPos, marioFloatPos))) {
            return 0; // Still
        }

        float dx = marioFloatPos[0] - prevPos[0];
        float dy = marioFloatPos[1] - prevPos[1];

        if (dx < 0 && dy < 0) {
            return 1;
        }
        if (dx < 0 && dy == 0) {
            return 2;
        }
        if (dx < 0 && dy > 0) {
            return 3;
        }
        if (dx == 0 && dy > 0) {
            return 4;
        }
        if (dx > 0 && dy > 0) {
            return 5;
        }
        if (dx > 0 && dy == 0) {
            return 6;
        }
        if (dx > 0 && dy < 0) {
            return 7;
        }
        if (dx == 0 && dy < 0) {
            return 8;
        }

        return 0;
    }

    private QState getCurrentState() {

        int[] s = new int[20];
        s[0] = marioState[0]; //
        s[1] = calculateDirection(); // Direction expressed as an integer
        s[2] = marioState[2]; // On ground?
        s[3] = marioState[3]; // Able to jump?
        s[4] = killed ? 1 : 0; // Killed last frame

        // more

        return new QState(s);

    }

    // TODO
    private QAction getRandomAction() {

        return new QAction();
    }

    // TODO
    private QAction getBestAction(QState state) {

        return new QAction();
    }

    // TODO
    @Override
    public boolean[] getAction() {

        QState state = getCurrentState();
        QAction action;

        // Should a random action be taken?
        if (random.nextFloat() < rho) {
            action = getRandomAction();
        } else {
            action = getBestAction(state);
        }

        // Pseudo code from Artificial Intelligence for Games

        // Q = getQValue(state, action)
        // maxQ =
        // Q = ( 1 - alpha) * Q + alpha * (reward + gamma * maxQ)
        // storeQValue(state, action, Q)
        // state = newState


        return action.actions;
    }

    // TODO
    @Override
    public void reset() {
        super.reset();
    }

    /**
     * Build the Q-table
     * TODO
     */
    @Override
    public void learn() {

        // Need to use task.evaluate(this) to train
        // Every evaluation updates the QTable
        int N = 2000;
        for (int i = 0; i < N; i++) {
            task.evaluate(this);
        }


    }

    @Override
    public void giveReward(float reward) {

    }

    // TODO
    @Override
    public void newEpisode() {
        task = null;
        reset();
    }

    @Override
    public void setLearningTask(LearningTask learningTask) {
        task = learningTask;
    }

    @Override
    public void setEvaluationQuota(long num) {
        evalQuota = num;
    }

    @Override
    public Agent getBestAgent() {
        return this;
    }

    @Override
    public void init() {

    }
}
