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
    private final int nrStates = 14;
    private final int nrActions = 12;
    private boolean debugEnabled = false;
    protected byte[][] enemies;
    private byte[][] mergedObservation;
    // -----------------------------------------------
    // Member variables

    protected float[] marioFloatPos;
    protected float[] enemiesFloatPos;
    protected int[] marioState = null;
    private int zLevelScene = 2;
    private int zLevelEnemies = 2;
    // Mario AI
    private int receptiveFieldWidth;
    private int receptiveFieldHeight;
    private int marioEgoRow;
    private int marioEgoCol;
    private byte[][] levelScene;
    private float[] prevMarioFloatPos;
    private float[] prevEnemiesFloatPos;
    private int killsWithStomp;
    private int prevKillsWithStomp;
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
    private int killsWithFire;
    private int prevKillsWithFire;
    private int prevElevation;
    private int marioX;
    private int prevMarioX;
    private boolean hasStarted;

    // -----------------------------------------------
    // Ctor
    public QLearningAgent() {
        setName("Q-Learning Agent");
        QTable = new HashMap<>();
        alpha0 = 0.8f;
        alpha = 0.15f;
        gamma = 0.62f;
        rho = 0.3f;

        actionList = buildKeyCombinations();

        reset();
    }


    private QState getCurrentState() {

        float dx = marioFloatPos[0] - prevMarioFloatPos[0];
        float dy = marioFloatPos[1] - prevMarioFloatPos[1];

        int[] s = new int[nrStates];
        s[0] = (marioState[1] == 0 || marioState[1] == 1) ? 1 : 0; // Small, big or fire
        s[1] = calculateDirection(dx, dy); // Direction expressed as an integer
        s[2] = marioState[2]; // On ground?
        s[3] = (marioState[3] == 1 || marioState[2] == 0) ? 1 : 0; // Able to jump?
        s[4] = nearbyDanger();
        s[5] = 0;
        s[6] = nearbyObstacle();
        s[7] = midrangeEnemies();
        s[8] = longRangeEnemies();
        s[9] = (killsTotal - prevKillsTotal) > 0 ? 1 : 0; // killed something previous frame

        int distance = marioX - prevMarioX;
        distance = Math.max(-2, Math.min(2, distance));
        s[10] = Math.abs(distance) < 2 ? 0 : distance / 2;
        s[11] = prevMarioMode > marioState[1] ? 1 : 0; // Took damage last frame?
        s[12] = (killsWithStomp - prevKillsWithStomp) > 0 ? 1 : 0;

        int elevation = Math.max(0, getDistanceToGround(marioEgoCol - 1) - getDistanceToGround(marioEgoCol));
        int dElevation = Math.max(0, elevation - prevElevation);
        s[13] = dElevation;

        prevElevation = elevation;

        return new QState(s);

    }

    private boolean isGround(int x, int y) {

        int val = filterObstacle(getReceptiveFieldCellValue(y, x));

        return val == 1;
    }

    private int getDistanceToGround(int x) {
        for (int y = marioEgoRow + 1; y < levelScene.length; y++) {
            if (isGround(x, y)) {
                return Math.min(3, y - marioEgoRow - 1);
            }
        }
        return -1;
    }

    // Reward function. Weighted value based on the input state.
    private float getReward(QState state) {

        float reward = 0;

        int rawState[] = state.states;
        boolean enemiesNearby = rawState[4] != -1;
        boolean enemiesMid = rawState[7] != -1;
        boolean enemiesFar = rawState[8] != -1;
        int distance = rawState[10];
        int elevation = rawState[13];
        boolean killedEnemy = rawState[9] == 1;
        boolean killedEnemyWithStomp = rawState[12] == 1;
//        boolean marioStuck = rawState[5] == 1;
        boolean marioTookDamage = rawState[11] == 1;

        float enemyScaler = 1;
        if (enemiesNearby || enemiesMid) {
            enemyScaler = 0f;
        } else if (enemiesFar) {
            enemyScaler = 0.15f;
        }

        reward += enemyScaler * distance * 2;
        reward += enemyScaler * elevation * 8;

        // Reward kills in general
        if (killedEnemy && !killedEnemyWithStomp) {
            reward += 60;
        }

//      Reward stomp kills extra
        if (killedEnemyWithStomp) {
            reward += 40;
        }

//      Punish being stuck
//        if (marioStuck) {
//            reward -= 20;
//        }

//      Punish getting hit
        if (marioTookDamage) {
//            System.out.println1"Took damage");
            reward -= 15000;
        }

        return reward;
    }

    private int calculateDirection(float dx, float dy) {

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

        if (Math.abs(dx) < 0.5) {
            dx = 0;
        }

        if (Math.abs(dy) < 0.5) {
            dy = 0;
        }

        float epsilon = 0.000001f;

        if (dx < -epsilon && dy > epsilon) {
            return 1;
        }
        if (dx < -epsilon && Math.abs(dy - 0.0f) < epsilon) {
            return 2;
        }
        if (dx < -epsilon && dy < -epsilon) {
            return 3;
        }
        if (Math.abs(dx - 0.0f) < epsilon && dy < -epsilon) {
            return 4;
        }
        if (dx > epsilon && dy < -epsilon) {
            return 5;
        }
        if (dx > epsilon && Math.abs(dy - 0.0f) < epsilon) {
            return 6;
        }
        if (dx > epsilon && dy > epsilon) {
            return 7;
        }
        if (Math.abs(dx - 0.0f) < epsilon && dy > epsilon) {
            return 8;
        }

        return 0;
    }

    private int nearbyDanger() {
        return checkEnemies(-1, 1, -1, 1);

    }

    private int midrangeEnemies() {

        return checkEnemies(-3, 3, -3, 3);
    }

    private int longRangeEnemies() {
        return checkEnemies(-5, 5, -5, 5);
    }

    private int checkEnemies(int x0, int x1, int y0, int y1) {

        int result = 0;
        boolean directions[] = new boolean[9];
        Arrays.fill(directions, false);

        y0 = marioState[1] == 1 ? y0 - 1 : y0;

        for (int i = x0; i <= x1; i++) {
            for (int j = y0; j <= y1; j++) {
                if (getEnemiesCellValue(marioEgoRow + j, marioEgoCol + i) != 0) {
                    int dir = calculateDirection(i, j);
                    if (!directions[dir]) {
                        result += Math.pow(2, dir);
                        directions[dir] = true;
                    }
                }
            }
        }

        return result == 0 ? -1 : result;

    }


    private boolean isStuck() {

        // Calculate delta distance
        double dx = marioFloatPos[0] - prevMarioFloatPos[0];
        float dy = marioFloatPos[1] - prevMarioFloatPos[1];

        if (Math.abs(dx) < 2.5f) {
            nrStuckFrames++;
        } else {
            nrStuckFrames = 0;
        }

        return nrStuckFrames > 24;

    }

    private int nearbyObstacle() {
        int ob1 = getReceptiveFieldCellValue(marioEgoRow - 2, marioEgoCol + 1);
        int ob2 = getReceptiveFieldCellValue(marioEgoRow - 1, marioEgoCol + 1);
        int ob3 = getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 1);
        int ob4 = getReceptiveFieldCellValue(marioEgoRow + 1, marioEgoCol + 1);

        ob1 = filterObstacle(ob1);
        ob2 = filterObstacle(ob2);
        ob3 = filterObstacle(ob3);
        ob4 = filterObstacle(ob4);

        return ob1 + 2 * ob2 + 4 * ob3 + 8 * ob4;
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
        boolean nothing[] = new boolean[6];

        boolean left[] = new boolean[6];
        left[Mario.KEY_LEFT] = true;
        boolean right[] = new boolean[6];
        right[Mario.KEY_RIGHT] = true;
        boolean jump[] = new boolean[6];
        jump[Mario.KEY_JUMP] = true;
        boolean fire[] = new boolean[6];
        fire[Mario.KEY_SPEED] = true;

        boolean left_jump[] = new boolean[6];
        left_jump[Mario.KEY_LEFT] = true;
        left_jump[Mario.KEY_JUMP] = true;
        boolean right_jump[] = new boolean[6];
        right_jump[Mario.KEY_RIGHT] = true;
        right_jump[Mario.KEY_JUMP] = true;

        boolean left_fire[] = new boolean[6];
        left_fire[Mario.KEY_LEFT] = true;
        left_fire[Mario.KEY_SPEED] = true;

        boolean right_fire[] = new boolean[6];
        right_fire[Mario.KEY_LEFT] = true;
        right_fire[Mario.KEY_SPEED] = true;

        boolean jump_fire[] = new boolean[6];
        jump_fire[Mario.KEY_JUMP] = true;
        jump_fire[Mario.KEY_SPEED] = true;

        boolean left_jump_fire[] = new boolean[6];
        left_jump_fire[Mario.KEY_JUMP] = true;
        left_jump_fire[Mario.KEY_SPEED] = true;
        left_jump_fire[Mario.KEY_LEFT] = true;

        boolean right_jump_fire[] = new boolean[6];
        right_jump_fire[Mario.KEY_JUMP] = true;
        right_jump_fire[Mario.KEY_SPEED] = true;
        right_jump_fire[Mario.KEY_RIGHT] = true;

        keyCombinations.add(nothing);
        keyCombinations.add(left);
        keyCombinations.add(right);
        keyCombinations.add(jump);
        keyCombinations.add(fire);
        keyCombinations.add(left_jump);
        keyCombinations.add(right_jump);
        keyCombinations.add(left_fire);
        keyCombinations.add(right_fire);
        keyCombinations.add(jump_fire);
        keyCombinations.add(left_jump_fire);
        keyCombinations.add(right_jump_fire);

        return keyCombinations;

    }


    @Override
    public void integrateObservation(Environment environment) {
        // Update environment information
        levelScene = environment.getLevelSceneObservationZ(zLevelScene);
        enemies = environment.getEnemiesObservationZ(zLevelEnemies);
        mergedObservation = environment.getMergedObservationZZ(zLevelScene, zLevelEnemies);

        this.marioFloatPos = environment.getMarioFloatPos();
        this.enemiesFloatPos = environment.getEnemiesFloatPos();
        this.marioState = environment.getMarioState();

        this.killsWithStomp = environment.getKillsByStomp();
        this.killsWithFire = environment.getKillsByFire();
        this.killsTotal = environment.getKillsTotal();

        this.marioX = environment.getEvaluationInfo().distancePassedPhys;

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
            int i = prevActions.nrTimesActionPerformed[prevActionIdx];
            float alphaTest = alpha0 / i;
            float newQ = (1 - alphaTest) * oldQ + alphaTest * (reward + gamma * maxQ);
            prevActions.setQ(prevActionIdx, newQ);
        }

        prevS = newState;
        prevActionIdx = actionIdx;
        prevMarioFloatPos = marioFloatPos.clone();
        prevEnemiesFloatPos = enemiesFloatPos.clone();
        prevKillsTotal = killsTotal;
        prevKillsWithStomp = killsWithStomp;
        prevKillsWithFire = killsWithFire;
        prevMarioMode = marioState[1];

        prevMarioX = marioX;


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

    @Override
    public boolean[] getAction() {
        if (prevActionIdx == -1) {
            return actionList.get(random.nextInt(nrActions));
        } else {
            return actionList.get(prevActionIdx);
        }
    }

    @Override
    public void reset() {

        hasStarted = false;

        marioX = 0;
        prevMarioX = 0;

        killsWithStomp = 0;
        killsWithFire = 0;

        killsTotal = 0;
        prevKillsTotal = 0;
        prevKillsWithStomp = 0;
        prevKillsWithFire = 0;
        prevElevation = 0;


        prevMarioFloatPos = new float[2];
        marioFloatPos = new float[2];
        marioFloatPos[0] = 0;
        marioFloatPos[1] = 0;
        prevMarioFloatPos[0] = 0;
        prevMarioFloatPos[1] = 0;
        prevMarioMode = 2; //
        nrStuckFrames = 0;

        prevActionIdx = -1;
        prevS = null;

    }

    //

    /**
     * Build the Q-table
     */
    @Override
    public void learn() {

        ArrayList<Float> intervalWinsPercentage = new ArrayList<>();
        int N = 5000;
        int totalScore = 0;
        int score = 0;
        int wins = 0;
        int lateWins = 0;
        int nDeaths = 0;
        int nTimesup = 0;
//        int avgDistanceCleared = 0;
        int iter = 0;
        int iterWins = 0;
        int interval = 500;
        for (int i = 1; i < N; i++, iter++) {
            score = task.evaluate(this);
            totalScore += score;
            EvaluationInfo eval = task.getEvaluationInfo();
            System.out.print("Iteration " + i + ", Score: " + score + ", QTable Size: " + QTable.size() + ", ");
            System.out.print("Cleared " + eval.distancePassedCells + " cells, ");
            if (eval.marioStatus == Mario.STATUS_WIN) {
                wins++;
                iterWins++;
                //System.out.print("Mario won!");
            } else if (eval.timeLeft == 0) {
                nTimesup++;
                //System.out.print("Time's up! Mario reached " + eval.distancePassedCells);
            } else {
                nDeaths++;
//                avgDistanceCleared += eval.distancePassedCells;
                //System.out.print("RIP Mario 2017. Time left: " + eval.timeLeft);
            }
            if (iter == interval) {
                intervalWinsPercentage.add((float) iterWins / (float) interval);
                iter = 0;
                iterWins = 0;
            }
            System.out.println("Wins: " + 100.f * (float) wins / (float) i
                    + "%, Deaths: " + 100.f * (float) nDeaths / (float) i
                    + "%, Times up: " + 100.f * (float) nTimesup / (float) i + "%");

        }

        System.out.println("Percentage won: " + (float) wins / (float) N);
        System.out.println("Percentage won late: " + (float) lateWins / (float) 100);
        System.out.println("Total score across all runs: " + totalScore);
//        System.out.println("Avg distance when timeout: " + avgDistanceCleared / nDeaths);

        for (float f : intervalWinsPercentage) {
            System.out.println(f);
        }


    }

    @Override
    public void giveReward(float reward) {

    }

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
//        alpha = 0;
        rho = 0.01f;
//        debugEnabled = true;
        return this;
    }

    @Override
    public void init() {
        reset();
    }

    private int getEnemiesCellValue(int x, int y) {
        if (x < 0 || x >= levelScene.length || y < 0 || y >= levelScene[0].length)
            return 0;

        return enemies[x][y];
    }

    private int getReceptiveFieldCellValue(int x, int y) {
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
            for (int i = 0; i < nrActions; i++) {
                qValues[i] = (random.nextFloat() * 2.f - 1.f);
            }
            nrTimesActionPerformed = new int[nrActions];
        }

        int getBestAction() {

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

        float getQ(int idx) {
            return qValues[idx];
        }


        void setQ(int idx, float newQ) {
            qValues[idx] = newQ;
        }


        int getRandomAction() {
            int rndIdx = random.nextInt(nrActions);
            nrTimesActionPerformed[rndIdx]++;
            return rndIdx;
        }

    }

}
