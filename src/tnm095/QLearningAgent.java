package tnm095;

import ch.idsia.agents.Agent;
import ch.idsia.agents.LearningAgent;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.benchmark.tasks.LearningTask;

import java.util.Arrays;
import java.util.Hashtable;
import java.util.Random;

/**
 * Created by Niclas Olmenius on 2017-09-25.
 */
public class QLearningAgent implements LearningAgent {


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
    private int receptiveFieldWidth;
    private int receptiveFieldHeight;
    private int marioEgoRow;
    private int marioEgoCol;

    int zLevelScene = 1;
    int zLevelEnemies = 0;

    private byte[][] levelScene;
    protected byte[][] enemies;
    protected byte[][] mergedObservation;

    protected float[] marioFloatPos = null;
    protected float[] enemiesFloatPos = null;

    protected int[] marioState = null;

    private float[] prevMarioFloatPos;
    private float[] prevEnemiesFloatPos;
    private boolean killed;
    private boolean killedWithFire;
    private boolean killedWithStomp;
    private boolean killedWithShell;
    private String name;

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

    QStateAction oldQStateAction;


    // -----------------------------------------------
    // Ctor
    public QLearningAgent() {
        setName("Q-Learning Agent");
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

    // TODO
    // Reward function. Weighted value based on the input state.
    private float getReward(QState state) {

        return 0.0f;
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

        if (prevMarioFloatPos == null || (Arrays.equals(prevMarioFloatPos, marioFloatPos))) {
            return 0; // Still
        }

        float dx = marioFloatPos[0] - prevMarioFloatPos[0];
        float dy = marioFloatPos[1] - prevMarioFloatPos[1];

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

        // Loop over every action for the input state
        // return the action that gives the highest Q-value

        return new QAction();
    }

    @Override
    public void integrateObservation(Environment environment) {
        levelScene = environment.getLevelSceneObservationZ(zLevelScene);
        enemies = environment.getEnemiesObservationZ(zLevelEnemies);
        mergedObservation = environment.getMergedObservationZZ(1, 0);

        this.marioFloatPos = environment.getMarioFloatPos();
        this.enemiesFloatPos = environment.getEnemiesFloatPos();
        this.marioState = environment.getMarioState();

        QState newState = getCurrentState(); // New state
        float reward = getReward(newState); //

        QAction action;

        // Should a random action be taken?
        if (random.nextFloat() < rho) {
            action = getRandomAction();
        } else {
            action = getBestAction(newState);
        }


        QStateAction qStateAction = new QStateAction(newState, action);

        QTable.putIfAbsent(qStateAction, 0.0f); // Add state to table if it is absent

        if (oldQStateAction != null) {

            float oldQ = QTable.get(oldQStateAction);
            float maxQ = QTable.get(qStateAction);
            float newQ = (1 - alpha) * oldQ + alpha * (reward + gamma * maxQ);

            QTable.replace(oldQStateAction, newQ);
        }

        oldQStateAction = qStateAction;
        prevMarioFloatPos = marioFloatPos;
        prevEnemiesFloatPos = enemiesFloatPos;

    }

    @Override
    public void giveIntermediateReward(float intermediateReward) {

    }

    @Override
    public void setObservationDetails(int rfWidth, int rfHeight, int egoRow, int egoCol) {
        receptiveFieldWidth = rfWidth;
        receptiveFieldHeight = rfHeight;

        marioEgoRow = egoRow;
        marioEgoCol = egoCol;
    }


    // TODO
    @Override
    public boolean[] getAction() {

        return oldQStateAction.qAction.actions;

    }

    // TODO
    @Override
    public void reset() {

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

    public int getEnemiesCellValue(int x, int y) {
        if (x < 0 || x >= levelScene.length || y < 0 || y >= levelScene[0].length)
            return 0;

        return enemies[x][y];
    }

    public int getReceptiveFieldCellValue(int x, int y) {
        if (x < 0 || x >= levelScene.length || y < 0 || y >= levelScene[0].length)
            return 0;

        return levelScene[x][y];
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }
}
