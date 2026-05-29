package tma.strategiesV2;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AIWithComputationBudget;
import rts.*;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;
import util.Pair;

import java.util.ArrayList;
import java.util.List;

public abstract class AwareAI extends AIWithComputationBudget{

    protected PathFinding pf;
    protected ScoutPathFinding spf = new ScoutPathFinding();

    public int getnHarvest() {
        return nHarvest;
    }

    public void setnHarvest(int nHarvest) {
        this.nHarvest = nHarvest;
    }

    public int getUnitAwareness() {
        return unitAwareness;
    }

    public void setUnitAwareness(int unitAwareness) {
        this.unitAwareness = unitAwareness;
    }

    public int getTotBarracks() {
        return totBarracks;
    }

    public void setTotBarracks(int totBarracks) {
        this.totBarracks = totBarracks;
    }

    public double getStrategyPriority() {
        return strategyPriority;
    }

    public void setStrategyPriority(double strategyPriority) {
        this.strategyPriority = strategyPriority;
    }

    public int[] getPlayerUnits() {
        return playerUnits;
    }

    public void setPlayerUnits(int[] playerUnits) {
        this.playerUnits = playerUnits;
    }

    public int[] getEnemyUnits() {
        return enemyUnits;
    }

    public void setEnemyUnits(int[] enemyUnits) {
        this.enemyUnits = enemyUnits;
    }

    public int[] getUnitProduction() {
        return unitProduction;
    }

    public void setUnitProduction(int[] unitProduction) {
        this.unitProduction = unitProduction;
    }

    public int getResourceTreshold() {
        return resourceTreshold;
    }

    public void setResourceTreshold(int resourceTreshold) {
        this.resourceTreshold = resourceTreshold;
    }

    int resourceTreshold = 5;

    protected int nHarvest = 2;
    int unitAwareness = 3;
    int totBarracks = 1;
    int heavyCounter = 0;

    double strategyPriority = 0.0;

    int[] playerUnits = new int[] {0, 0, 0, 0, 0, 0};
    int[] enemyUnits = new int[] {0, 0, 0, 0, 0, 0};
    int[] unitProduction = new int[] {1, 1, 1};

    protected UnitTypeTable utt;
    UnitType workerType;
    UnitType baseType;
    UnitType barracksType;
    UnitType rangedType;
    UnitType lightType;
    UnitType heavyType;

    PlayerAction pa;

    public AwareAI(PathFinding a_pf, int timeBudget, int iterationsBudget) {
        super(timeBudget, iterationsBudget);
        pf = a_pf;
    }

    public AwareAI(PathFinding a_pf){
        super(-1, -1);
        pf = a_pf;
    }

    public void reset() {
        //TODO
    }

    public void reset(UnitTypeTable a_utt)
    {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        rangedType = utt.getUnitType("Ranged");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
    }

    public PlayerAction getAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);
        pa = new PlayerAction();

        List<Unit> workers = new ArrayList<>();

        ResourceUsage ru = new ResourceUsage();
        ResourceUsage r = gs.getResourceUsage();
        pa.setResourceUsage(r);

        for(Unit u : gs.getUnits()){
            if (u.getPlayer() != player) {
                continue;
            }

            UnitAction a = null;

            if(u.getType() == baseType && gs.getActionAssignment(u) == null)
                a = baseBehavior(u, p, gs);

            if(u.getType() == barracksType && gs.getActionAssignment(u) == null)
                a = barracksBehavior(u, p, gs);

            if(u.getType().canHarvest)
                workers.add(u);

            if(u.getType().canAttack && !u.getType().canHarvest  && gs.getActionAssignment(u) == null){
                a = meleeUnitBehavior(u, p, gs, ru);
            }

            if(a != null){
                ResourceUsage r_a = a.resourceUsage(u, pgs);

                if(pa.consistentWith(r_a, gs)){
                    ru.merge(r_a);
                    pa.addUnitAction(u, a);
                    pa.getResourceUsage().merge(r_a);
                }
            }
        }

        List<Pair<Unit, UnitAction>> w_a = workersBehavior(workers, p, gs, ru);

        if(w_a != null || !w_a.isEmpty()){
            for(Pair<Unit, UnitAction> pair : w_a){
                if(pair.m_b != null){
                    ResourceUsage r_a = pair.m_b.resourceUsage(pair.m_a, pgs);

                    if(pa.consistentWith(r_a, gs)){
                        ru.merge(r_a);
                        pa.addUnitAction(pair.m_a, pair.m_b);
                        pa.getResourceUsage().merge(r_a);
                    }
                }
            }
        }

        pa.fillWithNones(gs, player, 10);
        return pa;
    }

    int manhattanDistance(int x, int y, int x2, int y2) {
        return Math.abs(x-x2) + Math.abs(y-y2);
    }

    double sqrDistance(int x, int y, int x2, int y2) {
        int dx = x - x2;
        int dy = y - y2;

        return Math.sqrt(dx * dx + dy * dy);
    }

    public int realDistance(Unit u1, Unit u2, GameState gs) {
        AStarPathFinding aStar = new AStarPathFinding();

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int d = aStar.findDistToPositionInRange(u1, u2.getPosition(pgs), 1, gs, gs.getResourceUsage());

        if(d == -1)
            return 9999;
        else
            return d;
    }

    public int posDistance(int a, int b, int w){
        return manhattanDistance(a%w, b%w, a/w, b/w);
    }

    public UnitAction Move(Unit unit, int dest, GameState gs, ResourceUsage ru){
        UnitAction move = pf.findPath(unit, dest, gs, ru);
//        System.out.println("AStarAttak returns: " + move);
        if (move!=null && gs.isUnitActionAllowed(unit, move))
            return move;
        if(move==null){
            //TODO: still get closer with other methods
        }
        return null;
    }

    public UnitAction Train(Unit unit, UnitType type, GameState gs){
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int x = unit.getX();
        int y = unit.getY();
        int best_direction = -1;
        int best_score = -1;

        if (y>0 && gs.free(x,y-1)) {
            int score = evaluateTrain(x,y-1, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_UP;
            }
        }
        if (x<pgs.getWidth()-1 && gs.free(x+1,y)) {
            int score = evaluateTrain(x+1,y, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_RIGHT;
            }
        }
        if (y<pgs.getHeight()-1 && gs.free(x,y+1)) {
            int score = evaluateTrain(x,y+1, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_DOWN;
            }
        }
        if (x>0 && gs.free(x-1,y)) {
            int score = evaluateTrain(x-1,y, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_LEFT;
            }
        }

        if (best_direction!=-1) {
            UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE,best_direction, type);
            if (gs.isUnitActionAllowed(unit, ua)) return ua;
        }

        return null;
    }

    public int evaluateTrain(int x, int y, UnitType type, int player, PhysicalGameState pgs) {
        int distance = 0;
        boolean first = true;

        if (type.canHarvest && playerUnits[0] < nHarvest) {
            // evaluateTrain is minus distance to closest resource
            for(Unit u:pgs.getUnits()) {
                if (u.getType().isResource) {
                    int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                    if (first || d<distance) {
                        distance = d;
                        first = false;
                    }
                }
            }
        } else {
            // evaluateTrain is minus distance to closest enemy
            for(Unit u:pgs.getUnits()) {
                if (u.getPlayer()>=0 && u.getPlayer()!=player) {
                    int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                    if (first || d<distance) {
                        distance = d;
                        first = false;
                    }
                }
            }
        }

        return -distance;
    }

    public UnitAction Build(Unit unit, UnitType type, int x, int y, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        UnitAction move = pf.findPathToAdjacentPosition(unit, x+y*pgs.getWidth(), gs, ru);
        if (move!=null) {
            if (gs.isUnitActionAllowed(unit, move)) return move;
            return null;
        }

        // build:
        UnitAction ua = null;
        if (x == unit.getX() &&
                y == unit.getY()-1) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_UP,type);
        if (x == unit.getX()+1 &&
                y == unit.getY()) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_RIGHT,type);
        if (x == unit.getX() &&
                y == unit.getY()+1) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_DOWN,type);
        if (x == unit.getX()-1 &&
                y == unit.getY()) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_LEFT,type);
        if (ua!=null && gs.isUnitActionAllowed(unit, ua)) return ua;

        return null;
    }

    public int evaluateBaseBuild(Unit builder, int pos, int player, PhysicalGameState pgs){
        int distance = 0, enemyDistance = 0, resourceDistance = 0;

        int x = pos % pgs.getWidth();
        int y = pos / pgs.getWidth();
        int posDistance = Math.abs(builder.getX() - x) + Math.abs(builder.getY() - y);

        for (Unit u : pgs.getUnits()){
            if (u.getType().isResource) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if((resourceDistance == 0 || d<resourceDistance) && d > 1){
                    resourceDistance = d;
                }
            }

            if (u.getPlayer()>=0 && u.getPlayer()!=player) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if (enemyDistance == 0 || d>enemyDistance) {
                    enemyDistance = d;
                }
            }
        }

        distance = enemyDistance - resourceDistance - posDistance;

        return distance;
    }

    public int evaluateBarrackBuild(int pos, int player, PhysicalGameState pgs){
        int distance = 0, enemyDistance = 0, baseDistance = 0;

        int x = pos % pgs.getWidth();
        int y = pos / pgs.getWidth();

        if(badSpot(x, y, pgs))
            return distance;

        for (Unit u : pgs.getUnits()){
            if (u.getType() == baseType && u.getPlayer() == player) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if((baseDistance == 0 || d>baseDistance) && d > 1){
                    baseDistance = d;
                }
            }

            if (u.getPlayer()>=0 && u.getPlayer()!=player) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if (enemyDistance == 0 || (d>enemyDistance && pgs.getWidth()<64) || (d<enemyDistance && pgs.getWidth()>=64)) {
                    enemyDistance = d;
                }
            }
        }

        if(pgs.getWidth()<64)
            distance = enemyDistance + baseDistance;
        else
            distance = baseDistance - enemyDistance;

        return distance;
    }

    public int evaluateAttackTarget(Unit unit, Unit target, PhysicalGameState pgs){
        int score = 0;
        int d = Math.abs(unit.getX() - target.getX()) + Math.abs(unit.getY() - target.getY());
        //int modifier = 0;
        if(d<=unit.getAttackRange() && oneShots(unit, target))
            return score;
        /*if(d<=4){
            switch (unit.getType().name){
                case "Worker":
                    if(target.getType()==workerType)
                        modifier -= 1;
                    break;
                case "Ranged":
                    if(target.getHitPoints()==1)
                        modifier -= unit.getAttackRange();
                    break;

                default:
                    break;
            }
        }*/
        //score = d + modifier;
        return d;
    }

    public boolean oneShots(Unit unit, Unit target){
        return target.getHitPoints()<=unit.getMaxDamage();
    }



    public List<Integer> GetFreePositionsAround(Unit u, PhysicalGameState pgs){
        List<Integer> positions = new ArrayList<>();
        int[] diffX = new int[] {-1, 0, +1, -1, +1, -1, 0, +1};
        int[] diffY = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};

        if(u.getX() == 0){
            diffX = new int[] {0, +1, +2, +1, +2, 0, +1, +2};
            diffY = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
        }
        else if(u.getX() == pgs.getWidth() - 1){
            diffX = new int[] {0, -1, -2, -1, -2, 0, -1, -2};
            diffY = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
        }
        else if(u.getY() == 0){
            diffX = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
            diffY = new int[] {0, +1, +2, +1, +2, 0, +1, +2};
        }
        else if(u.getY() == pgs.getHeight() - 1){
            diffX = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
            diffY = new int[] {0, -1, -2, -1, -2, 0, -1, -2};
        }

        for(int i = 0; i < diffX.length; i++){
            int x = u.getX() + diffX[i], y = u.getY() + diffY[i];

            if(x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight())
                continue;

            if(pgs.getUnitAt(x, y) == null && pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_WALL && !AdjacentToResource(x, y, pgs)){
                positions.add(x + y * pgs.getWidth());
            }
        }

        return positions;
    }

    public boolean AdjacentToResource(int x, int y, PhysicalGameState pgs){
        int[] diffX = new int[] {-1, 0, +1, 0};
        int[] diffY = new int[] {0, -1, 0, +1,};

        for(int i = 0; i < diffX.length; i++){
            int nx = x + diffX[i], ny = y + diffY[i];

            if(nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight())
                continue;

            Unit aU = pgs.getUnitAt(nx, ny);

            if(aU != null){
                if(aU.getType().isResource)
                    return true;
            }
        }

        return false;
    }

    //Checks if the selected spot is either near a Resource or on a chokepoint
    public boolean badSpot(int x, int y, PhysicalGameState pgs){
        int[] diffX = new int[] {-1, 0, +1, 0, -1, -1, +1, +1};
        int[] diffY = new int[] {0, -1, 0, +1, -1, +1, -1, +1};

        List<Integer> walls = new ArrayList<>();

        for(int i = 0; i < diffX.length; i++){
            int nx = x + diffX[i], ny = y + diffY[i];

            if(nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight())
                continue;

            Unit aU = pgs.getUnitAt(nx, ny);

            if(aU != null){
                if(aU.getType().isResource)
                    return true;
            }

            if(pgs.getTerrain(nx,ny) == PhysicalGameState.TERRAIN_WALL){
                walls.add(nx + ny * pgs.getWidth());
                if(walls.size() > 1){
                    int wall = walls.get(walls.size()-1);
                    for (int other: walls) {
                        if(posDistance(wall, other, pgs.getWidth()) > 1)
                            return true;
                    }
                }
            }

        }

        return false;
    }

    public boolean isChokepoint(int x, int y, PhysicalGameState pgs){
        int[] diffX = new int[] {-1, 0, +1, 0, -1, -1, +1, +1};
        int[] diffY = new int[] {0, -1, 0, +1, -1, +1, -1, +1};

        List<Integer> walls = new ArrayList<>();

        for(int i = 0; i < diffX.length; i++){
            int nx = x + diffX[i], ny = y + diffY[i];

            if(nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight())
                continue;

            if(pgs.getTerrain(nx,ny) == PhysicalGameState.TERRAIN_WALL){
                walls.add(nx + ny * pgs.getWidth());
                if(walls.size() > 1){
                    int wall = walls.get(walls.size()-1);
                    for (int other: walls) {
                        if(posDistance(wall, other, pgs.getWidth()) > 1)
                            return true;
                    }
                }
            }

        }

        return false;

    }

    public boolean isPlayerPredominant(Unit unit, int width, int height, GameState gs) {
        boolean mine;

        int allies = 0, enemies = 0;

        for (Unit u : gs.getUnits()) {
            if ((Math.abs(u.getX() - unit.getX()) <= width && Math.abs(u.getY() - unit.getY()) <= height)) {
                if(u.getPlayer() == unit.getPlayer())
                    allies++;

                if(u.getPlayer() != unit.getPlayer() && u.getPlayer() >= 0)
                    enemies++;
            }
        }

        if(enemies > allies)
            mine = false;
        else
            mine = true;

        return mine;
    }

    public boolean inRange(Unit u1, Unit u2, int range){
        return ((Math.abs(u1.getX() - u2.getX()) <= range && Math.abs(u1.getY() - u2.getY()) <= range));
    }

    public Unit GetClosestBase(Unit u, PhysicalGameState pgs){
        Unit closestBase = null;
        int distance = 0;

        for(Unit other : pgs.getUnits()){
            if(other.getType() == baseType && other.getPlayer() == u.getPlayer()){
                int d = Math.abs(u.getX() - other.getX()) + Math.abs(u.getY() - other.getY());
                if(closestBase == null || d < distance){
                    closestBase = other;
                    distance = d;
                }
            }
        }

        return closestBase;
    }

    public UnitAction Harvest(Unit unit, Unit target, Unit base, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        if (unit.getResources()==0) {
            if (target == null) return null;
            // go get resources:
            UnitAction move = pf.findPathToAdjacentPosition(unit, target.getX()+target.getY()*gs.getPhysicalGameState().getWidth(), gs, ru);
            if (move!=null) {
                if (gs.isUnitActionAllowed(unit, move)) return move;
                return null;
            }

            // harvest:
            if (target.getX() == unit.getX() &&
                    target.getY() == unit.getY()-1) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_UP);
            if (target.getX() == unit.getX()+1 &&
                    target.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_RIGHT);
            if (target.getX() == unit.getX() &&
                    target.getY() == unit.getY()+1) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_DOWN);
            if (target.getX() == unit.getX()-1 &&
                    target.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_LEFT);
        } else {
            // return resources:
            if (base == null) return null;
            UnitAction move = pf.findPathToAdjacentPosition(unit, base.getX()+base.getY()*gs.getPhysicalGameState().getWidth(), gs, ru);
            if (move!=null) {
                if (gs.isUnitActionAllowed(unit, move)) return move;
                return null;
            }

            // harvest:
            if (base.getX() == unit.getX() &&
                    base.getY() == unit.getY()-1) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_UP);
            if (base.getX() == unit.getX()+1 &&
                    base.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_RIGHT);
            if (base.getX() == unit.getX() &&
                    base.getY() == unit.getY()+1) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_DOWN);
            if (base.getX() == unit.getX()-1 &&
                    base.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_LEFT);
        }
        return null;
    }

    public UnitAction Attack(Unit unit, Unit target, GameState gs, ResourceUsage ru) {
        if(target == null)
            return null;

        if (inAttackRange(unit, target) && !willEscapeAttack(unit, target, gs)) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,target.getX(),target.getY());
        } else if(futureInAttackRange(unit, target, gs)){
            return new UnitAction(UnitAction.TYPE_NONE, 1);
        }  else {
            UnitAction move = pf.findPathToPositionInRange(unit, target.getX()+target.getY()*gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
            if (move!=null && gs.isUnitActionAllowed(unit, move))
                return move;

            if(move == null)
                return Approach(unit, target, gs, ru);

            return null;
        }
    }

    protected UnitAction rangedBehavior(Unit ranged, Player p, GameState gs, ResourceUsage ru){
        Unit closestEnemy = null;
        Unit bestTarget = null;
        Unit closestAlly = null;
        int closestDistance = 0;
        int bestTargetScore = 0;
        int allyDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = manhattanDistance(ranged.getX(), ranged.getY(), u2.getX(), u2.getY());
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }

                int t = evaluateAttackTarget(ranged, u2, pgs);
                if (bestTarget==null || t<bestTargetScore) {
                    bestTarget = u2;
                    bestTargetScore = t;
                }
            }
            else if(u2.getPlayer()==p.getID()){
                int l = manhattanDistance(ranged.getX(), ranged.getY(), u2.getX(), u2.getY());
                if (closestAlly==null || l<allyDistance) {
                    closestAlly = u2;
                    allyDistance = l;
                }
            }
        }
        if(bestTarget!=null && closestEnemy!=null){
            if(closestDistance<=2){
                if(oneShots(ranged, bestTarget))
                    return Attack(ranged, bestTarget, gs, ru);

                int f_target = futurePos(closestEnemy, gs);

                int f_x = f_target%pgs.getWidth();
                int f_y = f_target/pgs.getWidth();
                int f_d = manhattanDistance(ranged.getX(), ranged.getY(), f_x, f_y);

                if(f_d == 1){
                    List<Integer> positions = PositionsInRadius(ranged, 1, pgs);
                    if(positions.isEmpty())
                        return Attack(ranged, bestTarget, gs, ru);
                    for(int pos : positions){
                        if(manhattanDistance(closestEnemy.getX(), closestEnemy.getY(), pos%pgs.getWidth(), pos/pgs.getWidth()) > f_d)
                            return Move(ranged, pos, gs, ru);
                    }
                }
                return Attack(ranged, bestTarget, gs, ru);
            }
            else
                return Attack(ranged, bestTarget, gs, ru);
        }
        else
            return Attack(ranged,null,gs,ru);
    }

    protected UnitAction aggroMeleeBehevior(Unit unit, Player p, GameState gs, ResourceUsage ru){
        Unit closestTarget = null;
        Unit favoriteTarget = null;
        int closestDistance = 0;
        int favoriteDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(unit, u2, pgs);
                if (closestTarget==null || d<closestDistance) {
                    closestTarget = u2;
                    closestDistance = d;
                }
                if(((u2.getType()==barracksType || u2.getType()==baseType) && unit.getType()==heavyType) ||
                        (u2.getType()==rangedType && unit.getType()==lightType)){
                    if (favoriteTarget==null || d<favoriteDistance) {
                        favoriteTarget = u2;
                        favoriteDistance = d;
                    }
                }
            }
        }
        if(favoriteTarget==null && closestTarget!=null)
            return Attack(unit,closestTarget,gs,ru);
        if(favoriteTarget==closestTarget)
            return Attack(unit, favoriteTarget, gs, ru);
        if(closestTarget!=null && closestDistance < 4)
            return Attack(unit,closestTarget,gs,ru);
        if(favoriteTarget!=null)
            return Attack(unit, favoriteTarget, gs, ru);
        return Attack(unit,null,gs,ru);
    }

    int combatScore(Unit u, Unit e, PhysicalGameState pgs) {
        int score = manhattanDistance(u.getX(), u.getY(), e.getX(), e.getY());

        if ((u.getType() == rangedType || u.getType() == lightType) && e.getType() == rangedType && pgs.getWidth() > 9)
            score -= 2;

        if (pgs.getWidth() >= 16 && (u.getType() == heavyType || u.getType() == rangedType)
                && (e.getType() == barracksType)) //todo - remove? todo base
            score -= pgs.getWidth();

        return score;
    }

    public UnitAction Scout(Unit unit, Unit target, GameState gs, ResourceUsage ru){
        if(target == null)
            return null;
        if (inAttackRange(unit, target) && !willEscapeAttack(unit, target, gs)) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,target.getX(),target.getY());
        } else if(futureInAttackRange(unit, target, gs)){
            return new UnitAction(UnitAction.TYPE_NONE, 1);
        }
        else{
            UnitAction move = spf.findPathToPositionInRange(unit, target.getX()+target.getY()*gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
            if (move!=null && gs.isUnitActionAllowed(unit, move)){
                if(move.getType() == UnitAction.TYPE_HARVEST && isChokepoint(unit.getX(), unit.getY(), gs.getPhysicalGameState()) && !isPlayerPredominant(unit, 5, 5, gs)){
                    return null;
                }
                else
                    return move;
            }


            if(move == null)
                return Approach(unit, target, gs, ru);

            return null;
        }
    }

    public UnitAction Idle(Unit unit, GameState gs, ResourceUsage ru){
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if (!unit.getType().canAttack)
            return null;

        for(Unit target:pgs.getUnits()) {
            if (target.getPlayer()!=-1 && target.getPlayer()!=unit.getPlayer()) {
                if (inAttackRange(unit, target)) {
                    return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,target.getX(),target.getY());
                }
            }
        }
        return null;
    }

    public UnitAction Approach(Unit unit, Unit target, GameState gs, ResourceUsage ru){
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Integer> positions = PositionsInRadius(unit, 2, pgs);
        int pos = 0;
        int closestDistance = 0;
        int modifier = 0;

        if(target.getType().canAttack){
            if(target.getType()==rangedType)
                modifier = 4;
            else
                modifier = 1;
        }

        int currentDistance = Math.abs(target.getX() - unit.getX()) + Math.abs(target.getY() - unit.getY()) - modifier;
        boolean found = false;

        if(positions.isEmpty())
            return null;

        if(target.getType().canAttack){
            if(target.getType()==rangedType)
                modifier = 4;
            else
                modifier = 1;
        }

        for(int p : positions){
            if(pf.pathExists(unit, p, gs, ru)){
                int x = p % pgs.getWidth();
                int y = p/ pgs.getWidth();
                int d = Math.abs(target.getX() - x) + Math.abs(target.getY() - y) - modifier;

                if((!found||d<closestDistance) && d < currentDistance && d > 0){
                    closestDistance = d;
                    pos = p;
                    found = true;
                }
            }
        }

        if(found){
            UnitAction move = pf.findPath(unit, pos, gs, ru);
            if(move!=null && gs.isUnitActionAllowed(unit, move))
                return move;
        }

        return null;
    }

    public List<Integer> PositionsInRadius(Unit u, int radius, PhysicalGameState pgs){
        List<Integer> positions = new ArrayList<>();
        int[] diffX = new int[] {-1, 0, +1, 0};
        int[] diffY = new int[] {0, -1, 0, +1,};

        if(radius == 2){
            diffX = new int[] {-1, 0, +1, 0, -1, -1, +1, +1, -2, 0, +2, 0};
            diffY = new int[] {0, -1, 0, +1, -1, +1, -1, +1, 0, -2, 0, +2};
        }

        for(int i = 0; i < diffX.length; i++){
            int x = u.getX() + diffX[i], y = u.getY() + diffY[i];

            if(x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight())
                continue;

            if(pgs.getUnitAt(x, y) == null && pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_WALL && !AdjacentToResource(x, y, pgs)){
                positions.add(x + y * pgs.getWidth());
            }
        }

        return positions;
    }

    public boolean Between(Unit unit, Unit first, Unit last){
        int minX = Math.min(first.getX(), last.getX());
        int maxX = Math.max(first.getX(), last.getX());

        int minY = Math.min(first.getY(), last.getY());
        int maxY = Math.max(first.getY(), last.getY());

        return (minX <= unit.getX() && unit.getX() <= maxX && minY <= unit.getY() && unit.getY() <= maxY);
    }

    public List<Unit> getAdjacentUnits(Unit u, PhysicalGameState pgs, boolean friendly){
        List<Unit> units = new ArrayList<>();
        int[] diffX = new int[] {-1, 0, +1, 0};
        int[] diffY = new int[] {0, -1, 0, +1,};

        for(int i = 0; i < diffX.length; i++){
            int x = u.getX() + diffX[i], y = u.getY() + diffY[i];

            if(x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight())
                continue;

            Unit u2 = pgs.getUnitAt(x, y);

            if(u2 != null){
                if(u2.getType().canAttack && u2.getPlayer() == u.getPlayer() && friendly)
                    units.add(u2);

                if(u2.getType().canAttack && u2.getPlayer() != u.getPlayer() && !friendly)
                    units.add(u2);
            }
        }

        return units;
    }

    public boolean conflictingMove(Unit unit, UnitAction unitAction, List<Pair<Unit, UnitAction>> list, PhysicalGameState pgs){
        List<Unit> adjacentUnits = getAdjacentUnits(unit, pgs, true);

        if(unitAction == null)
            return false;

        if(unitAction.getType()!=UnitAction.TYPE_MOVE)
            return false;

        if(!adjacentUnits.isEmpty()){
            for (Unit au: adjacentUnits) {
                for(Pair<Unit, UnitAction> pair : list){
                    if(pair.m_b==null)
                        continue;

                    if(pair.m_a == au && pair.m_b.getType() == UnitAction.TYPE_MOVE)
                        return true;
                }
            }
        }

        return false;
    }

    int futurePos(Unit unit, GameState gs){
        int x = unit.getX(), y = unit.getY();
        PhysicalGameState pgs = gs.getPhysicalGameState();

        UnitActionAssignment aa = gs.getActionAssignment(unit);
        if (aa == null){
            return x + y * pgs.getWidth();
        }

        else if (aa.action.getType() == UnitAction.TYPE_MOVE){
            int nx = x;
            int ny = y;

            switch (aa.action.getDirection()) {
                case UnitAction.DIRECTION_DOWN:
                    ny = (ny == pgs.getHeight()- 1) ? ny : ny + 1;
                    break;
                case UnitAction.DIRECTION_UP:
                    ny = (ny == 0) ? ny : ny - 1;
                    break;
                case UnitAction.DIRECTION_RIGHT:
                    nx = (nx == pgs.getWidth() - 1) ? nx : nx + 1;
                    break;
                case UnitAction.DIRECTION_LEFT:
                    nx = (nx == 0) ? nx : nx - 1;
                    break;
                default:
                    break;
            }

            return nx + ny * pgs.getWidth();
        }

        return x + y * pgs.getWidth();
    }

    boolean futureInAttackRange(Unit u1, Unit u2, GameState gs){
        int futurePos = futurePos(u2, gs);

        int w = gs.getPhysicalGameState().getWidth();
        int x2 = futurePos%w;
        int y2 = futurePos/w;

        boolean inRange = sqrDistance(u1.getX(), u1.getY(), x2, y2) <= u1.getAttackRange();

        return inRange;
    }

    boolean inAttackRange(Unit u1, Unit u2) {
        return sqrDistance(u1.getX(), u1.getY(), u2.getX(), u2.getY()) <= u1.getAttackRange();
    }

    boolean willEscapeAttack(Unit attacker, Unit runner, GameState gs) {
        UnitActionAssignment aa = gs.getActionAssignment(runner);
        if (aa == null)
            return false;
        if (aa.action.getType() != UnitAction.TYPE_MOVE)
            return false;
        int eta = aa.action.ETA(runner) - (gs.getTime() - aa.time);
        return eta <= attacker.getAttackTime();
    }

    protected abstract UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru);
    protected abstract UnitAction barracksBehavior(Unit u, Player p, GameState gs);
    protected abstract UnitAction baseBehavior(Unit u, Player p, GameState gs);
    protected abstract UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru);
    protected abstract UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru);
    protected abstract UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru);
    protected abstract List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru);
}