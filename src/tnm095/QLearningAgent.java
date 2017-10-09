package tnm095;

import ch.idsia.agents.Agent;
import ch.idsia.agents.LearningAgent;
import ch.idsia.benchmark.mario.engine.GeneralizerLevelScene;
import ch.idsia.benchmark.mario.engine.sprites.Mario;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.benchmark.tasks.LearningTask;
import ch.idsia.tools.EvaluationInfo;

import java.util.*;

/**
 * Created by Niclas Olmenius on 2017-09-25.
 */
public class QLearningAgent implements LearningAgent {


    // Q-learning
    public static final Random random = new Random();
    private final int nrStates = 10;
    protected byte[][] enemies;
    protected byte[][] mergedObservation;
    // -----------------------------------------------
    // Member variables
    protected float[] marioFloatPos;
    protected float[] enemiesFloatPos;
    protected int[] marioState = null;
    int zLevelScene = 2;
    int zLevelEnemies = 2;
    QState currState;
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
    private int prevKillsTotal;
    private LearningTask task;
    private long evalQuota;
    private ArrayList<QAction> keyCombinations;
    private HashMap<QState, QActions> QTable;
    private QState prevS;
    private QAction prevA;
    private float oldQ;
    private float prevR;
    private float alpha;
    private float gamma;
    private float rho;

    // -----------------------------------------------
    // Ctor
    public QLearningAgent() {
        setName("Q-Learning Agent");
        QTable = new HashMap<>(20000);
        alpha = 0.15f;
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

        prevS = new QState(new int[nrStates]);
        prevA = new QAction(new boolean[32]);

        keyCombinations = buildKeyCombinations();

        reset();
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
            reward += 0.7;
        } else if (dir == 1 || dir == 2 || dir == 3) { // moving backwards
            reward += -0.3;
        }

        // Reward kills
        if (rawState[9] == 1) {
            reward += 1.f;
        }

        // Punish being stuck
        if (rawState[5] == 1) {
            reward += -0.4;
        }


        return reward * 20;
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
    }

    private int midrangeEnemies() {

        int nearEnemies = 0;
        for (int i = 2; i < 3; i++) {
            for (int j = -4; j < 3; j++) {
                nearEnemies = nearEnemies * getEnemiesCellValue(marioEgoRow + i, marioEgoCol + j);
            }
        }
        return nearEnemies;
    }

    private int longRangeEnemies() {
        int longRangeEnemies = 0;
        for (int i = 4; i < 5; i++) {
            for (int j = -6; j < 5; j++) {
                longRangeEnemies = longRangeEnemies * getEnemiesCellValue(marioEgoRow + i, marioEgoCol + j);
            }
        }
        return longRangeEnemies;
    }

    private boolean isStuck() {

        // Calculate delta distance
        float dx = marioFloatPos[0] - prevMarioFloatPos[0];
        float dy = marioFloatPos[1] - prevMarioFloatPos[1];

        return (dx * dx + dy * dy < 0.001f);
    }

    private int nearbyObstacle() {
        int ob1 = getReceptiveFieldCellValue(marioEgoRow + 2, marioEgoCol + 1);
        int ob2 = getReceptiveFieldCellValue(marioEgoRow + 1, marioEgoCol + 1);
        int ob3 = getReceptiveFieldCellValue(marioEgoRow, marioEgoCol + 1);
        int ob4 = getReceptiveFieldCellValue(marioEgoRow - 1, marioEgoCol + 1);

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

    private QState getCurrentState() {

        int[] s = new int[nrStates];
        s[0] = marioState[1]; // Small, big or fire
        s[1] = calculateDirection(); // Direction expressed as an integer
        s[2] = marioState[2]; // On ground?
        s[3] = marioState[3]; // Able to jump?
//        s[4] = getEnemiesCellValue(marioEgoRow, marioEgoCol);
        s[4] = nearbyDanger() ? 1 : 0;
        s[5] = isStuck() ? 1 : 0;
        s[6] = nearbyObstacle();
        s[7] = midrangeEnemies();
        s[8] = longRangeEnemies();
        s[9] = (killsTotal - prevKillsTotal) > 0 ? 1 : 0; // killed something previous frame


        return new QState(s);

    }

    private ArrayList<QAction> buildKeyCombinations() {
        ArrayList<QAction> keyCombinations = new ArrayList<>();

        int n = Environment.numberOfKeys;
        int m = 32;

        for (int x = 0; x < m; x++) {
            boolean[] b = new boolean[n - 1];
            for (int i = 0; i < n - 1; i++) b[i] = (1 << n - 1 - i - 1 & x) != 0;
            boolean[] b2 = Arrays.copyOf(b, 6);
            keyCombinations.add(new QAction(b2));
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


//        if (QTable.size() > 100) {
//            System.out.println("test");
//        }

        float reward = getReward(prevS);

        QActions actions;
        if (!QTable.containsKey(newState)) {
            actions = new QActions();
            QTable.put(newState, actions);
        } else {
            actions = QTable.get(newState);
        }

        QAction newAction;
        // Should a random action be taken?
        if (random.nextFloat() < rho) {
            newAction = actions.getRandomAction();
        } else {
            newAction = actions.getBestAction();
        }


        float maxQ = actions.getQ(newAction);
        float newQ = (1 - alpha) * oldQ + alpha * (reward + gamma * maxQ);
        QActions prevActions = QTable.get(prevS);
        prevA.Q = newQ;
        prevActions.replaceAction(prevA);

        QTable.replace(prevS, prevActions);


        oldQ = newQ;
        prevS = newState;
        prevA = newAction;
        prevMarioFloatPos = marioFloatPos.clone();
        prevEnemiesFloatPos = enemiesFloatPos.clone();
        prevKillsTotal = killsTotal;

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

        return prevA.actions.clone();

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
        int N = 15000;
        int totalScore = 0;
        int score = 0;
        int wins = 0;
        int lateWins = 0;
        int nDeaths = 0;
        int nTimesup = 0;
        for (int i = 0; i < N; i++) {
//            alpha = 0.1f + 0.6f * (float) Math.exp(-i * 2.71f / N);
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
                //System.out.print("RIP Mario 2017. Time left: " + eval.timeLeft);
            }
            System.out.println("Wins: " + 100.f * (float) wins / (float) (i + 1) + "%, Deaths: " + 100.f * (float) nDeaths / (float) (i + 1) + "%, Times up: " + 100.f * (float) nTimesup / (float) (i + 1) + "%");

        }

        System.out.println("Percentage won: " + (float) wins / (float) N);
        System.out.println("Percentage won late: " + (float) lateWins / (float) 500);

        System.out.println(QTable.size());


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
//        alpha = 0;
//        rho = 0;
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

        public QState(QState newState) {
            states = newState.states.clone();
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

        ArrayList<QAction> actions;
        //        boolean actions[];
        int actionIndex;

        float[] qValues;

        QActions() {
            actions = buildKeyCombinations();
        }

        public QActions(QActions qActions) {
            actions = (ArrayList<QAction>) qActions.actions.clone();
        }

        public QAction getBestAction() {

            float maxQ = Float.NEGATIVE_INFINITY;
            float Q;
            int bestIdx = 0;
            for (int i = 0; i < 32; i++) {
                Q = actions.get(i).Q;
                if (Q > maxQ) {
                    maxQ = Q;
                    bestIdx = i;
                }
            }
            return new QAction(actions.get(bestIdx));
        }

        public float getQ(QAction action) {
            return actions.get(actions.indexOf(action)).Q;
        }


        public void replaceAction(QAction action) {
            actions.set(actions.indexOf(action), action);
        }

        public QAction getRandomAction() {
            return new QAction(actions.get(random.nextInt(32)));
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QActions qActions = (QActions) o;

            return actions.equals(qActions.actions);
        }

        @Override
        public int hashCode() {
            return actions.hashCode();
        }
    }

    private class QAction {
        boolean[] actions;
        float Q;

        QAction(boolean[] a) {
            actions = a;
            Q = 0;
        }

        public QAction(QAction action) {
            actions = action.actions.clone();
            Q = action.Q;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QAction qAction = (QAction) o;

            return Arrays.equals(actions, qAction.actions);
        }

        @Override
        public int hashCode() {
            int result = Arrays.hashCode(actions);
            return result;
        }
    }

//    private class QStateAction {
//        QState qState;
//        QAction qAction;
//
//        QStateAction(QState q, QAction a) {
//            qState = q;
//            qAction = a;
//        }
//
//        @Override
//        public boolean equals(Object o) {
//            if (this == o) return true;
//            if (o == null || getClass() != o.getClass()) return false;
//
//            QStateAction that = (QStateAction) o;
//
//            if (!qState.equals(that.qState)) return false;
//            return qAction.equals(that.qAction);
//        }
//
//        @Override
//        public int hashCode() {
//            int result = qState.hashCode();
//            result = 31 * result + qAction.hashCode();
//            return result;
//        }
//    }
}
