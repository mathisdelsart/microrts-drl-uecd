package tma.strategiesV2;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import rts.*;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import util.Pair;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class AWorkDefense extends AwareAI{
    public AWorkDefense(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public AWorkDefense(UnitTypeTable a_utt){
        this(a_utt, new AStarPathFinding());
    }

    @Override
    public AI clone() {
        return new AWorkDefense(utt, pf);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return null;
    }

    @Override
    protected UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestBase = null;
        Unit closestResource = null;
        int closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestResource==null || d<closestDistance) {
                    closestResource = u2;
                    closestDistance = d;
                }
            }
        }
        closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestBase==null || d<closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
        }
        if(closestBase!=null||closestResource!=null)
            return Harvest(worker, closestResource, closestBase, gs, ru);

        return scoutBehavior(worker, p, gs, ru);
    }

    @Override
    protected UnitAction barracksBehavior(Unit u, Player p, GameState gs) {
        if (p.getResources()>=lightType.cost)
            return Train(u, lightType, gs);
        return null;
    }

    @Override
    protected UnitAction baseBehavior(Unit u, Player p, GameState gs) {
        if (p.getResources()>=workerType.cost)
            return Train(u, workerType, gs);
        return null;
    }

    @Override
    protected UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit closestMeleeEnemy = null;
        int closestDistance = 0;
        int enemyDistance = 0;
        int mybase = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        int threshold = Math.max(pgs.getHeight(), pgs.getWidth());

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(u, u2, pgs);
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
            else if(u2.getPlayer()==p.getID() && u2.getType() == baseType){
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if(mybase == 0 || d < mybase)
                    mybase = d;
            }
        }
        if (closestEnemy!=null && (closestDistance < threshold/2 || mybase < threshold/2)) {
            return Attack(u,closestEnemy,gs,ru);
        }
        else
        {
            return Attack(u,null,gs,ru);
        }
    }

    @Override
    protected UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit base = null;
        int closestEnemyDistance = 0;
        int closestBaseDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        int threshold = Math.max(pgs.getHeight(), pgs.getWidth());

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(scout, u2, pgs);
                if (closestEnemy==null || d<closestEnemyDistance) {
                    closestEnemy = u2;
                    closestEnemyDistance = d;
                }
            }
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - scout.getX()) + Math.abs(u2.getY() - scout.getY());
                if (base==null || d<closestBaseDistance) {
                    base = u2;
                    closestBaseDistance = d;
                }
            }
        }

        if(closestEnemy!=null && (closestEnemyDistance < threshold/2 || closestBaseDistance < threshold/2)){
            if(pf.findPathToAdjacentPosition(scout, closestEnemy.getPosition(pgs), gs, ru)!=null
                    || manhattanDistance(scout.getX(), scout.getY(), closestEnemy.getX(), closestEnemy.getY()) == 1){
                if(isChokepoint(scout.getX(), scout.getY(), pgs)){
                    //might make them stall regardless, could look into it
                    if(isPlayerPredominant(scout, 5, 5, gs))
                        return Attack(scout, closestEnemy, gs, ru);
                    else
                        return Idle(scout, gs, ru);
                }
                else
                    return Attack(scout, closestEnemy, gs, ru);
            }
            else{
                if(scout.getResources() == 0)
                    return Scout(scout,closestEnemy,gs,ru);
                else if (base!=null){
                    if(pf.findPathToAdjacentPosition(scout, base.getPosition(pgs), gs, ru)!=null
                            || manhattanDistance(scout.getX(), scout.getY(), base.getX(), base.getY()) == 1)
                        return Harvest(scout, null, base, gs, ru);
                    else
                        return Scout(scout,closestEnemy,gs,ru);
                }
            }
        }else
        {
            return Attack(scout,null,gs,ru);
        }

        return Scout(scout,null,gs,ru);
    }

    @Override
    protected UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if(playerUnits[4] == 0 && p.getResources() >= baseType.cost){
            int highscore = 0, pos = 0;

            List<Integer> positions = GetFreePositionsAround(worker, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBaseBuild(worker, i, p.getID(), pgs);
                if(score > highscore){
                    highscore = score;
                    pos = i;
                }
            }

            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, baseType, x, y, gs, ru);
        }

        return workerBehavior(worker, p, gs, ru);
    }

    @Override
    protected List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru) {
        int nbases = 0;
        int nbarracks = 0;
        int resourcesUsed = 0;
        Unit hw;
        UnitAction unitAction;
        List<Unit> harvestWorkers = new LinkedList<>();
        List<Unit> freeWorkers = new LinkedList<>(workers);
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Pair<Unit, UnitAction>> list = new ArrayList<>();

        if (workers.isEmpty()) return list;

        nbases = playerUnits[4];
        nbarracks = playerUnits[5];

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases==0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources()>=baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                //buildIfNotAlreadyBuilding(u,baseType,u.getX(),u.getY(),reservedPositions,p,pgs);
                if(gs.getActionAssignment(u) == null){
                    resourcesUsed+=baseType.cost;
                    unitAction = builderBehavior(u, p, gs, ru);

                    if(conflictingMove(u, unitAction, list, pgs))
                        unitAction = Idle(u, gs, ru);

                    list.add(new Pair<>(u, unitAction));
                }
            }
        }

        if (freeWorkers.size()>0){
            for(int i = 1; i <=nHarvest; i++){
                if(!freeWorkers.isEmpty()){
                    hw = freeWorkers.remove(0);
                    harvestWorkers.add(hw);
                }
            }
        }

        // harvest with the harvest worker:
        for (Unit harvestWorker : harvestWorkers) {
            if(gs.getActionAssignment(harvestWorker) != null)
                continue;

            Unit closestBase = null;
            Unit closestResource = null;
            Unit closestEnemy = null;

            int closestResourceDistance = 0;
            int closestBaseDistance = 0;
            int closestEnemyDistance = 0;

            for(Unit u2:pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestResource==null || d<closestResourceDistance) {
                        closestResource = u2;
                        closestResourceDistance = d;
                    }
                }

                if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestBase==null || d<closestBaseDistance) {
                        closestBase = u2;
                        closestBaseDistance = d;
                    }
                }

                if(u2.getPlayer() >= 0 && u2.getPlayer()!=harvestWorker.getPlayer()){
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestEnemy==null || d<closestEnemyDistance) {
                        closestEnemy = u2;
                        closestEnemyDistance = d;
                    }
                }

            }

            //if you are close to a base and an enemy is close to that, attack him
            if(closestEnemy != null && closestEnemyDistance <= 2 && closestBaseDistance <= 2)
                list.add(new Pair<>(harvestWorker, Attack(harvestWorker, closestEnemy, gs, ru)));

            else if (closestResource!=null && closestBase!=null) {
                if(closestEnemy!=null){
                    if(Between(closestEnemy, harvestWorker, closestResource) && harvestWorker.getResources()<1){
                        //this could be changed directly to Attack
                        unitAction = scoutBehavior(harvestWorker, p, gs, ru);
                    }
                    else
                        unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);
                }
                else
                    unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else if(closestBase!=null && harvestWorker.getResources()>0){
                unitAction = Harvest(harvestWorker, null, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else
                freeWorkers.add(harvestWorker);
        }

        for(Unit u:freeWorkers){
            if(gs.getActionAssignment(u) == null){
                unitAction = scoutBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }

        return list;
    }
}