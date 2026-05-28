package tma;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import tma.strategiesV2.*;
import rts.*;
import rts.units.Unit;
import rts.units.UnitTypeTable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TMA extends AIWithComputationBudget {

    UnitTypeTable localUtt;

    //height is enough since most of maps are squares
    boolean initialized = false;
    int mapMaxSize;
    int baseToResources = 0;
    int enemyToPlayerBase = 0;
    int baseToEnemyBase = 9999;
    int playerToEnemyBase = 0;
    int realBaseToEnemy = 0;
    int resourceTreshold = 5;
    double priority = 6.0;
    int awareness = 3;

    //In order: Workers, Light, Heavy, Ranged, Bases, Barracks
    int[] playerUnits = new int[] {0, 0, 0, 0, 0, 0};
    int[] enemyUnits = new int[] {0, 0, 0, 0, 0, 0};

    //First value determines either Attack or Defence (0 is defence, 1 is attack)
    //Second value determines either Workers of Military (could be extended to Ranged/Light/Heavy)
    AwareAI strategies[][];

    //Hardcoded values obtained by testing and tracking performance
    //Future improvement: weights that calibrate themselves through ML
    int[] weightAD = new int[] {1, 1};
    int[] weightWM = new int[] {1, 10, 1};

    Unit mainBase = null;
    Unit enemyBase = null;
    Unit closestEnemy = null;

    AStarPathFinding aStar = new AStarPathFinding();

    public TMA(AwareAI[][] s, int timeBudget, int iterationsBudget, UnitTypeTable utt) {
        super(timeBudget, iterationsBudget);
        localUtt = utt;
        strategies = s;
    }

    //called by Micro-RTS
    public TMA(UnitTypeTable utt){

        this(new AwareAI[][]{
                {new AWorkDefense(utt), new AMilDefense(utt)}, //defence strategies
                {new AWorkRush(utt), new AMilRush(utt)}//attack strategies
            },100, -1, utt);
    }

    @Override
    public void reset() { }

    @Override
    public AI clone() {
        return new TMA(strategies, TIME_BUDGET, ITERATIONS_BUDGET, localUtt);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("TimeBudget", int.class, 100));
        parameters.add(new ParameterSpecification("IterationsBudget", int.class, -1));
        return parameters;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if(!initialized){
            mapMaxSize = Math.max(gs.getPhysicalGameState().getHeight(), gs.getPhysicalGameState().getWidth());
            //calibrateStrategyPriority(mapMaxSize);
            updateUnitDistribution(player, gs);
            calibrateWeights(gs);
            initialized = true;
        }

        if (!gs.canExecuteAnyAction(player))
            return new PlayerAction();

        PlayerAction pa = new PlayerAction();
        AwareAI macroStrategy = getMacroStrategy(player, gs);
        pa = macroStrategy.getAction(player, gs);

        return pa;

    }

    private void calibrateStrategyPriority(int mapMaxSize){
        if(mapMaxSize <= 8){
            priority = 10.0;
        }
        else if (mapMaxSize > 8 && mapMaxSize<=16){
            priority = 5.0;
        }
        else if(mapMaxSize>16 && mapMaxSize<=32){
            priority = 3.0;
        }
        else{
            priority = 2.0;
        }
    }

    public AwareAI getMacroStrategy(int player, GameState gs){
        int a, b;
        double tmpA, tmpB;

        int x0, x1, y0, y1, y2;

        AwareAI strategy;
        int resources = gs.getPlayer(player).getResources();
        PhysicalGameState pgs = gs.getPhysicalGameState();
        updateUnitDistribution(player, gs);

        if (gs.getTime() % 500 == 0 && gs.getTime() != 0) {
            calibrateWeights(gs);
        }


        if(mainBase != null && realBaseToEnemy <= 0)
            realBaseToEnemy = distRealUnitEneBase(mainBase, gs.getPlayer(player), gs);

        if(mainBase!=null && enemyBase!=null){
            baseToEnemyBase = Math.abs(enemyBase.getX() - mainBase.getX()) + Math.abs(enemyBase.getY() - mainBase.getY());
        }

        Unit closestResource = null, closestPlayer = null;
        int rDistance = 9999, eDistance = 9999, pDistance = 9999, d;

        for(Unit u : pgs.getUnits()){
            if(mainBase!=null){
                if (u.getType().isResource) {
                    d = Math.abs(u.getX() - mainBase.getX()) + Math.abs(u.getY() - mainBase.getY());
                    if (closestResource==null || d<rDistance) {
                        closestResource = u;
                        rDistance = d;
                    }
                }
                else if (u.getPlayer() != player && u.getPlayer()>=0){
                    d = Math.abs(u.getX() - mainBase.getX()) + Math.abs(u.getY() - mainBase.getY());
                    if (closestEnemy==null || d<eDistance) {
                        closestEnemy = u;
                        eDistance = d;
                    }
                }
            }

            if(enemyBase!=null){
                if (u.getPlayer() == player){
                    d = Math.abs(u.getX() - enemyBase.getX()) + Math.abs(u.getY() - enemyBase.getY());
                    if (closestPlayer==null || d<pDistance) {
                        closestPlayer = u;
                        pDistance = d;
                    }
                }
            }

        }

        baseToResources = rDistance;
        enemyToPlayerBase = eDistance;
        playerToEnemyBase = pDistance;

        x0 = Arrays.stream(Arrays.copyOfRange(playerUnits, 0, 4)).sum() - Arrays.stream(Arrays.copyOfRange(enemyUnits, 0, 4)).sum();
        if(enemyToPlayerBase == 9999)
            x1 = 0;
        else if(playerToEnemyBase == 9999)
            x1 = 10;
        else
            x1 = enemyToPlayerBase - playerToEnemyBase;

        tmpA = x0 * weightAD[0] + x1 * weightAD[1];

        if(tmpA >= 0)
            a = 1;
        else
            a = 0;

        y0 = resources - 5; //hardcoded value, we might look into that

        if(playerUnits[5] > 0)
            y1 = 2;
        else
            y1 = 0;

        if(realBaseToEnemy == -1){
            y2 = 10;
        }
        else
            y2 = Math.min(enemyToPlayerBase, pgs.getHeight() + pgs.getWidth()) - 16; //hardcoded value, we might look into that

        tmpB = y0 * weightWM[0] + y1 * weightWM[1] + y2 * weightWM[2];


        if(tmpB >= 0)
            b = 1;
        else
            b = 0;

        strategy = strategies[a][b];
        CalibrateStrategy(strategy, gs, player);
        return strategy;
    }

    //This is called to set decision points inside the called AwareAI to adapt it to the current game situation
    public void CalibrateStrategy(AwareAI s, GameState gs, int player){
        int harvestUnits = 2;
        int nb = 1;

        if(mainBase == null)
            harvestUnits = 0;
        else if ((mapMaxSize < 12 || baseToEnemyBase < 16))
            harvestUnits = playerUnits[4];
        else if(baseToResources > enemyToPlayerBase) harvestUnits = Math.max(2, playerUnits[4]);
        else if(baseToResources/5 > harvestUnits)
            harvestUnits = Math.max(baseToResources/5, playerUnits[4]);
        else if(mapMaxSize >= 64)
            harvestUnits = Math.min(baseToResources - 1, 4);

        s.setnHarvest(harvestUnits);

        //sets the unit production
        //gives priority to the unit with higher number
        int[] unitProduction = CalibrateUnitProduction(gs);
        s.setUnitProduction(unitProduction);

        /*if(mapMaxSize>=64)
            nb = 2;
        else*/
        nb = 1;

        s.setTotBarracks(nb);

        if(mapMaxSize>=64 && realBaseToEnemy!=-1){
            resourceTreshold = 10;
        }
        else if(playerUnits[5] < nb){
            if(mapMaxSize<=10)
                resourceTreshold = 5 * (nb - playerUnits[5]);
            else
                resourceTreshold = 8 * (nb - playerUnits[5]);
        }
        else{
            if(mapMaxSize<=10)
                resourceTreshold = 3 * playerUnits[5];
            else
                resourceTreshold = 4 * playerUnits[5];
        }

        s.setResourceTreshold(resourceTreshold);
        s.setPlayerUnits(playerUnits);
        s.setEnemyUnits(enemyUnits);
        s.setUnitAwareness(awareness);
        s.setStrategyPriority(priority);
    }

    public int[] CalibrateUnitProduction(GameState gs){
        int[] units = new int[] {0, 0, 0};

        //aims for balanced unit distribution by default
        int playerHighest = Math.max(playerUnits[1], Math.max(playerUnits[2], playerUnits[3]));
        if(playerHighest == 0)
            units = new int[] {0, 1, 0};
        else{
            for(int i = 0; i < units.length; i++){
                if(playerUnits[i + 1] < playerHighest)
                    units[i] = 1;
            }
        }

        //whenever we can't reach the enemy directly, Ranged units are preferred
        boolean trainRanged = false;

        if(closestEnemy!=null){
            if(mainBase!=null){
                int d = aStar.findDistToPositionInRange(mainBase, closestEnemy.getPosition(gs.getPhysicalGameState()), 1, gs, gs.getResourceUsage());

                if(d==-1)
                    trainRanged = true;
            }
        }

        if(realBaseToEnemy == -1 || (baseToEnemyBase <= 4 && playerUnits[3] == 0) || trainRanged)
            units[2] = units[2] + 3;

        //adapt to enemy
        int toCounter = 0, enemyHighest = 0;

        for(int i = 1; i <= 3; i++){
            if(enemyUnits[i] > enemyHighest && enemyUnits[i] >= playerUnits[i]){
                toCounter = i;
                enemyHighest = enemyUnits[i];
            }
        }

        switch (toCounter){
            case 1://light
                units[1] = units[1] + 1;
                break;
            case 2://heavy
                units[1] = units[1] + 1;
                break;
            case 3://ranged
                units[0] = units[0] + 1;
                break;
        }

        System.out.println(units[0] + " " + units[1] + " " + units[2]);

        return units;
    }

    public void updateUnitDistribution(int player, GameState gs){
        playerUnits = new int[] {0, 0, 0, 0, 0, 0};
        enemyUnits = new int[] {0, 0, 0, 0, 0, 0};
        int un = 0;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        for(Unit u: pgs.getUnits()){

            if(u.getType().name=="Worker")
                un = 0;
            if(u.getType().name=="Light")
                un = 1;
            if(u.getType().name == "Heavy")
                un = 2;
            if(u.getType().name == "Ranged")
                un = 3;
            if(u.getType().name == "Base"){
                un = 4;
                if((mainBase == null) && u.getPlayer() == player)
                    mainBase = u;
                else if ((enemyBase == null) && u.getPlayer() != player)
                    enemyBase = u;
            }

            if(u.getType().name == "Barracks")
                un = 5;

            if(u.getPlayer() != player && u.getPlayer() >= 0)
                enemyUnits[un]++;
            else if(u.getPlayer() == player)
                playerUnits[un]++;
        }

        if(playerUnits[4] == 0)
            mainBase = null;

        if(enemyUnits[4] == 0)
            enemyBase = null;
    }

    public int distRealUnitEneBase(Unit base, Player p, GameState gs) {
        if(base == null)
            return 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestEnemy = null;
        int closestDistance = 0;
        int d = 9999;
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getPlayer() >= 0 && u2.getPlayer() != p.getID()) {
                if (u2 != null) {
                    d = aStar.findDistToPositionInRange(base, u2.getPosition(pgs), 1, gs, gs.getResourceUsage());
                    if ((closestEnemy == null || d < closestDistance) && d >= 0) {
                        closestEnemy = u2;
                        closestDistance = d;
                    }
                }

            }
        }
        if (closestEnemy == null) {
            return -1;
        } else {
            return closestDistance;
        }
    }

    public void calibrateWeights(GameState gs){
        if(playerUnits[0] == 0 && playerUnits[4] == 0){
            weightAD = new int[] {0, 0};
            weightWM = new int[] {0, 0, 0};
        }

        else{
            weightAD = new int[] {1, 1};
            weightWM = new int[] {1, 10, 1};
        }
    }
}