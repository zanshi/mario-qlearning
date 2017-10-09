package tnm095;

import ch.idsia.agents.Agent;
import ch.idsia.agents.LearningAgent;
import ch.idsia.benchmark.mario.engine.GeneralizerLevelScene;
import ch.idsia.benchmark.mario.engine.sprites.Mario;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.benchmark.tasks.LearningTask;
import ch.idsia.tools.EvaluationInfo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by Niclas Olmenius on 2017-09-25.
 */
public class QLearningAgent implements LearningAgent {

    // Q-learning
    public static final Random random = new Random();
    private final int nrStates = 12;
    private final int nrActions = 32;
    protected byte[][] enemies;
    protected byte[][] mergedObservation;
    // -----------------------------------------------
    // Member variables

    protected float[] marioFloatPos;
    protected float[] enemiesFloatPos;
    protected int[] marioState = null;
    int zLevelScene = 2;
    int zLevelEnemies = 2;
    //    QStateAction oldQStateAction;
    // Mario AI
    private int receptiveFieldWidth;
    private int receptiveFieldHeight;
    private int marioEgoRow;
    private int marioEgoCol;
    private byte[][] levelScene;
    private float[] prevMarioFloatPos;
    private float[] prevEnemiesFloatPos;
    private boolean killedWithFire;
    private boolean killedWithStomp;
    private boolean killedWithShell;
    private String name;
    private int killsTotal;
    private int prevMarioMode;
    private int prevKillsTotal;
    private LearningTask task;
    private long evalQuota;
    private ArrayList<boolean[]> actionList;
    private HashMap<QState, QActions> QTable;
    private QState prevS;
    private float alpha;
    private float alpha0;
    private float gamma;
    private float rho;
    private int prevActionIdx;
    private int nrStuckFrames;

    // -----------------------------------------------
    // Ctor
    public QLearningAgent() {
        setName("Q-Learning Agent");
        QTable = new HashMap<>();
        alpha0 = 0.8f;
        alpha = 0.10f;
        gamma = 0.6f;
        rho = 0.3f;

        killedWithFire = false;
        killedWithStomp = false;
        killedWithShell = false;

        prevKillsTotal = 0;

        prevMarioFloatPos = new float[2];
        marioFloatPos = new float[2];
        marioFloatPos[0] = 0;
        marioFloatPos[1] = 0;
        prevMarioFloatPos[0] = 0;
        prevMarioFloatPos[1] = 0;
        prevMarioMode = 2; //
        nrStuckFrames = 0;

        actionList = buildKeyCombinations();
        prevActionIdx = -1;

        reset();
    }


    private QState getCurrentState() {

        int[] s = new int[nrStates];
        s[0] = marioState[1]; // Small, big or fire
        s[1] = calculateDirection(); // Direction expressed as an integer
        s[2] = marioState[2]; // On ground?
        s[3] = marioState[3]; // Able to jump?
        s[4] = nearbyDanger() ? 1 : 0;
        s[5] = isStuck() ? 1 : 0;
//        s[5] = 0;
        s[6] = nearbyObstacle();
        s[7] = midrangeEnemies();
        s[8] = longRangeEnemies();
        s[9] = (killsTotal - prevKillsTotal) > 0 ? 1 : 0; // killed something previous frame

        float dx = marioFloatPos[0] - prevMarioFloatPos[0];
        float dy = marioFloatPos[1] - prevMarioFloatPos[1];
        if (dx > 2.2f) {
            s[10] = 2; // Going fast
        } else if (dx > 1.5f) {
            s[10] = 1; // Medium speed
        } else {
            s[10] = 0; // Slow
        }
        s[11] = prevMarioMode != marioState[1] ? 1 : 0; // Took damage last frame?

        return new QState(s);

    }

    // TODO: Tweak weights
    // Reward function. Weighted value based on the input state.
    private float getReward(QState state) {


//        int[] s = new int[10];
//        s[0] = marioState[1]; // Small, big or fire
//        s[1] = calculateDirection(); // Direction expressed as an integer
//        s[2] = marioState[2]; // On ground?
//        s[3] = marioState[3]; // Able to jump?
//        s[4] = getEnemiesCellValue(marioEgoRow, marioEgoCol);
//        s[5] = nearbyDanger() ? 1 : 0;
//        s[6] = isStuck() ? 1 : 0;
//        s[7] = nearbyObstacle();
//        s[8] = midrangeEnemies();
//        s[9] = longRangeEnemies();
//        s[10] = (killsTotal - prevKillsTotal) > 0 ? 1 : 0; // killed something previous frame

        float reward = 0;

        int rawState[] = state.states;

        int dir = rawState[1];
        if (dir == 5 || dir == 6 || dir == 7) { // moving forward
            reward += 0.6;
        } else if (dir == 0 || dir == 1 || dir == 2 || dir == 3 || dir == 4) { // moving backwards
            reward += -0.3;
        }

        // Reward kills
        if (rawState[9] == 1) {
            reward += 3.f;
        }

        // Punish being stuck
        if (rawState[5] == 1) {
//            System.out.println("stuck!");
            reward += -1.0f;
        }

        // Punish getting hit
        if (rawState[11] == 1) {
//            System.out.println("Took damage");
            reward += -5.0f;
        }

        if (rawState[4] != 1) {
//            System.out.println("no nearby enemies");
            // Reward going fast
            if (rawState[10] == 2) {
                reward += 0.8;
            } else if (rawState[10] == 1) {
                reward += 0.5;
            }
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

    private boolean nearbyDanger() {
        return (getReceptiveFieldCellValue(marioEgoRow + 2, marioEgoCol + 1) == 0 &&
                getReceptiveFieldCellValue(marioEgoRow + 1, marioEgoCol + 1) == 0) ||
                getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 1) != 0 ||
                getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 2) != 0 ||
                getEnemiesCellValue(marioEgoRow, marioEgoCol + 1) != 0 ||
                getEnemiesCellValue(marioEgoRow, marioEgoCol + 2) != 0;
//                getEnemiesCellValue(marioEgoRow - 1, marioEgoCol + 1) != 0 ||
//                getEnemiesCellValue(marioEgoRow - 1, marioEgoCol + 2) != 0 ||
//                getEnemiesCellValue(marioEgoRow - 1, marioEgoCol) != 0;
    }

    private int midrangeEnemies() {

        for (int i = 2; i < 3; i++) {
            for (int j = -4; j < 3; j++) {
                if (getEnemiesCellValue(marioEgoRow + i, marioEgoCol + j) == 1) {
                    return 1;
                }
            }
        }
        return 0;
    }

    private int longRangeEnemies() {
        for (int i = 4; i < 5; i++) {
            for (int j = -6; j < 5; j++) {
                if (getEnemiesCellValue(marioEgoRow + i, marioEgoCol + j) == 1) {
                    return 1;
                }
            }
        }
        return 0;
    }

    private boolean isStuck() {

        // Calculate delta distance
        double dx = marioFloatPos[0] - prevMarioFloatPos[0];
//        System.out.println(dx);
//        System.out.println(marioFloatPos[0]);
//        System.out.println(prevMarioFloatPos[0]);
        float dy = marioFloatPos[1] - prevMarioFloatPos[1];

//        return (dx * dx + dy * dy < 0.001f);
        if ((dx * dx) < 0.001f) {
            nrStuckFrames++;
        } else {
            nrStuckFrames = 0;
        }

        return nrStuckFrames > 3;
    }

    private int nearbyObstacle() {
        int ob1 = getReceptiveFieldCellValue(marioEgoRow - 2, marioEgoCol + 1);
        int ob2 = getReceptiveFieldCellValue(marioEgoRow - 1, marioEgoCol + 1);
        int ob3 = getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 1);
        int ob4 = getReceptiveFieldCellValue(marioEgoRow + 1, marioEgoCol + 1);
        int ob5 = getReceptiveFieldCellValue(marioEgoRow + 2, marioEgoCol + 1);


        ob1 = filterObstacle(ob1);
        ob2 = filterObstacle(ob2);
        ob3 = filterObstacle(ob3);
        ob4 = filterObstacle(ob4);
        ob5 = filterObstacle(ob5);

//        System.out.println("Test");
//        System.out.println(ob1);
//        System.out.println(ob2);
//        System.out.println(ob3);
//        System.out.println(ob4);

        return ob1 + 2 * ob2 + 4 * ob3 + 8 * ob4 + 16 * ob5;
    }

    private int filterObstacle(int obj) {

        if (obj == GeneralizerLevelScene.COIN_ANIM || obj == GeneralizerLevelScene.PRINCESS) {
            return 0;
        } else if (obj != 0) {
            return 1;
        }
        return obj;
    }


    private ArrayList<boolean[]> buildKeyCombinations() {
        ArrayList<boolean[]> keyCombinations = new ArrayList<>();

        int n = Environment.numberOfKeys;

        for (int x = 0; x < nrActions; x++) {
            boolean[] b = new boolean[n - 1];
            for (int i = 0; i < n - 1; i++) b[i] = (1 << n - 1 - i - 1 & x) != 0;
            boolean[] b2 = Arrays.copyOf(b, 6);
            keyCombinations.add(b2);
        }

        return keyCombinations;

    }


    // TODO
    @Override
    public void integrateObservation(Environment environment) {
        // Update environment information
        levelScene = environment.getLevelSceneObservationZ(zLevelScene);
        enemies = environment.getEnemiesObservationZ(zLevelEnemies);
        mergedObservation = environment.getMergedObservationZZ(zLevelScene, zLevelEnemies);

        this.marioFloatPos = environment.getMarioFloatPos();
        this.enemiesFloatPos = environment.getEnemiesFloatPos();
        this.marioState = environment.getMarioState();


        killsTotal = environment.getKillsTotal();

        // ----------------
        // Q-learning

        QState newState = getCurrentState(); // New state

        if (prevS == null) {
            prevS = newState;
        }

        float reward = getReward(prevS);

        QActions actions;
        if (!QTable.containsKey(newState)) {
            QTable.put(newState, new QActions());
        }
        actions = QTable.get(newState);

        int actionIdx;
        // Should a random action be taken?
        if (random.nextFloat() < rho) {
            actionIdx = actions.getRandomAction();
        } else {
            actionIdx = actions.getBestAction();
        }

        if (prevActionIdx != -1) {
            QActions prevActions = QTable.get(prevS);
            float oldQ = prevActions.getQ(prevActionIdx);
            float maxQ = actions.getQ(actionIdx);
            float alphaTest = alpha0 / prevActions.nrTimesActionPerformed[prevActionIdx];
            float newQ = (1 - alphaTest) * oldQ + alphaTest * (reward + gamma * maxQ);
            prevActions.setQ(prevActionIdx, newQ);
        }

        prevS = newState;
        prevActionIdx = actionIdx;
        prevMarioFloatPos = marioFloatPos.clone();
        prevEnemiesFloatPos = enemiesFloatPos.clone();
        prevKillsTotal = killsTotal;
        prevMarioMode = marioState[1];

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
        if (prevActionIdx == -1) {
            return actionList.get(random.nextInt(nrActions));
        } else {
            return actionList.get(prevActionIdx);
        }
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
        int N = 3000;
        int totalScore = 0;
        int score = 0;
        int wins = 0;
        int lateWins = 0;
        int nDeaths = 0;
        int nTimesup = 0;
        int iterations = 0;
        for (int i = 0; i < N; i++) {
            float scale = (float) Math.exp(-i * Math.E / N);
//            alpha = 0.1f + 0.7f * scale;
//            rho = 0.1f + 0.2f * scale;
            totalScore += task.evaluate(this);
            //alpha = 0.1f + 0.6f * (float) Math.exp(-i);
            EvaluationInfo eval = task.getEvaluationInfo();
            score = eval.computeWeightedFitness();
            System.out.print("Iteration " + i + ", Score: " + score + ", QTable Size: " + QTable.size() + ", ");
            if (eval.marioStatus == Mario.STATUS_WIN) {
                wins++;
                if (i > 1500) {
                    lateWins++;
                }
                //System.out.print("Mario won!");
            } else if (eval.timeLeft == 0) {
                nTimesup++;
                //System.out.print("Time's up! Mario reached " + eval.distancePassedCells);
            } else {
                nDeaths++;
//                giveReward(-5.0f);
                //System.out.print("RIP Mario 2017. Time left: " + eval.timeLeft);
            }
            System.out.println("Wins: " + 100.f * (float) wins / (float) (i + 1)
                    + "%, Deaths: " + 100.f * (float) nDeaths / (float) (i + 1)
                    + "%, Times up: " + 100.f * (float) nTimesup / (float) (i + 1) + "%");

        }

        System.out.println("Percentage won: " + (float) wins / (float) N);
//        System.out.println("Percentage won late: " + (float) lateWins / (float) 500);


    }

    @Override
    public void giveReward(float reward) {
//        System.out.println("give reward");
//

        QActions prevActions = QTable.get(prevS);
        prevActions.setQ(prevActionIdx, reward);
//        QState newState = getCurrentState(); // New state
//
//        if (prevS == null) {
//            prevS = newState;
//        }
//
//        QActions actions;
//        if (!QTable.containsKey(newState)) {
//            QTable.put(newState, new QActions());
//        }
//        actions = QTable.get(newState);
//
//        int actionIdx;
//        // Should a random action be taken?
//        if (random.nextFloat() < rho) {
//            actionIdx = actions.getRandomAction();
//        } else {
//            actionIdx = actions.getBestAction();
//        }
//
//        if (prevActionIdx != -1) {
//            QActions prevActions = QTable.get(prevS);
//            float oldQ = prevActions.getQ(prevActionIdx);
//            float maxQ = actions.getQ(actionIdx);
////            float alphaTest = alpha0 / prevActions.nrTimesActionPerformed[prevActionIdx];
//            float alpha = 0.7f;
//            float newQ = (1 - alpha) * oldQ + alpha * (reward + gamma * maxQ);
//            prevActions.setQ(prevActionIdx, newQ);
//        }
//
//        prevS = newState;
//        prevActionIdx = actionIdx;
//        prevMarioFloatPos = marioFloatPos.clone();
//        prevEnemiesFloatPos = enemiesFloatPos.clone();
//        prevKillsTotal = killsTotal;
//        prevMarioMode = marioState[1];

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
//        alpha0 = 0;
        rho = 0.01f;
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

    // Relevant data structures
    private class QState {
        int[] states;

        QState(int[] s) {
            states = s;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QState qState = (QState) o;

            return Arrays.equals(states, qState.states);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(states);
        }
    }

    private class QActions {

        float[] qValues;
        int[] nrTimesActionPerformed;

        QActions() {
            qValues = new float[nrActions];
//            for (int i = 0; i < nrActions; i++) {
//                qValues[i] = (0.1f * random.nextFloat() - 0.05f) * 2.0f;
//            }
            nrTimesActionPerformed = new int[nrActions];
        }

        public int getBestAction() {

            float maxQ = Float.NEGATIVE_INFINITY;
            float Q;
            int bestIdx = -1;
            for (int i = 0; i < nrActions; i++) {
                Q = qValues[i];
                if (Q > maxQ) {
                    maxQ = Q;
                    bestIdx = i;
                }
            }
            nrTimesActionPerformed[bestIdx]++;
            return bestIdx;
        }

        public float getQ(int idx) {
            return qValues[idx];
        }


        public void setQ(int idx, float newQ) {
            qValues[idx] = newQ;
        }


        public int getRandomAction() {
            int rndIdx = random.nextInt(nrActions);
            nrTimesActionPerformed[rndIdx]++;
            return rndIdx;
        }

    }

}
