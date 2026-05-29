/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mayariBot;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import static rts.PhysicalGameState.TERRAIN_WALL;
import rts.Player;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

/**
 *
 * @author uv
 * 
 * 
 * version 2.0
*/
public class Mayari extends AIWithComputationBudget {
        
    public class Pos {
        int _x;
        int _y;
        Pos(int x, int y) {
            _x = x;
            _y = y;
        }
        public int getX() { return _x; }
        public int getY() { return _y; }
    }
    
    UnitTypeTable _utt = null;
    
    AStarPathFinding _astarPath;
    
    int NoDirection = 100; //this is a hack
    long _startCycleMilli;
    long _latestTsMilli;
    
    PlayerAction _pa;
    
    private GameState _gs;
    private PhysicalGameState _pgs;
    Player _p;
    Player _enemyP;
    
    List<Integer> _dirs;
    
    List<Long> _memHarvesters;
    
    List<Integer> _locationsTaken; //x+y*width
    
    int _resourcesUsed;
    List<Pos> _futureBarracks;
    int _futureHeavies;
    int _enemyFutureHeavy;
    
    List<Unit> _resources;
    
    List<Unit> _bases;
    List<Unit> _barracks;
    List<Unit> _workers;
    List<Unit> _heavies;
    List<Unit> _archers;
    List<Unit> _lights;
    List<Unit> _allyUnits;
    List<Unit> _allyCombat;

    List<Unit> _enemyBases;
    List<Unit> _enemyBarracks;
    List<Unit> _enemyWorkers;
    List<Unit> _enemyHeavies;
    List<Unit> _enemyArchers;
    List<Unit> _enemyLights;
    List<Unit> _enemies;
    List<Unit> _enemiesCombat;

    List<Unit> _all;    
    HashMap<Unit, Integer> _newDmgs;

    public void restartPathFind() {
        _astarPath = new AStarPathFinding();
    }
    
    
    boolean isBlocked(Unit u, Pos p) {
        if (outOfBound(p) || _pgs.getTerrain(p.getX(), p.getY()) != PhysicalGameState.TERRAIN_NONE)
            return true;
        if (!posFree(p.getX(), p.getY(), NoDirection))
            return true;
        Unit pu = _pgs.getUnitAt(p.getX(), p.getY());
        if (pu == null)
            return false;
        if (pu.getType().isResource)
            return true;
        if (!isEnemyUnit(pu))
            return true;
        if (u.getType() == _utt.getUnitType("Worker") 
                && pu.getType() != _utt.getUnitType("Worker"))
            return true;
        return false;
    }
    
    UnitAction findPath(Unit u, Pos dst, int maxDist) {
        int proximity[][] = new int[_pgs.getWidth()][_pgs.getHeight()];
        for (int[] row: proximity)
            Arrays.fill(row, Integer.MAX_VALUE);
        proximity[dst.getX()][dst.getY()] = 0;
        int dist = 1;
        List<Pos> markNext = allPosDist(dst, 1);
        while (!markNext.isEmpty() && dist <= maxDist) {
            List<Pos> queue = new ArrayList<>();
            for (Pos p : markNext) {
                if (isBlocked(u, p) || proximity[p.getX()][p.getY()] != Integer.MAX_VALUE)
                    continue;
                proximity[p.getX()][p.getY()] = dist;
                List<Pos> nn = allPosDist(p, 1);
                for (Pos n : nn) {
                    if (isBlocked(u, n) || proximity[n.getX()][n.getY()] != Integer.MAX_VALUE || queue.contains(n))
                        continue;
                    queue.add(n);
                }
            }
            if (proximity[u.getX()][u.getY()] != Integer.MAX_VALUE)
                break;
            dist += 1;
            markNext.clear();
            markNext.addAll(queue);
        }
        //now lets see if there is a path
        List<Pos> moves = allPosDist(toPos(u), 1);
        Integer bestFit = Integer.MIN_VALUE;
        Pos bestPos = null;
        for (Pos p : moves) {
            if (outOfBound(p) || _pgs.getTerrain(p.getX(), p.getY()) == TERRAIN_WALL)
                continue;
            if (proximity[p.getX()][p.getY()] == Integer.MAX_VALUE)
                continue;
            Unit pu = _pgs.getUnitAt(p.getX(), p.getY());
            if (pu != null)
                continue;
            int fit = -1000*proximity[p.getX()][p.getY()] - (int)squareDist(p, dst);
            if (fit > bestFit) {
                bestFit = fit;
                bestPos = p;
            }
        }
        if (bestPos == null)
            return null;
        int dir = toDir(toPos(u), bestPos);
        return new UnitAction(UnitAction.TYPE_MOVE, dir);
    }
    
    UnitAction findPathAdjacent(Unit src, Integer dst) {
        int x = dst % _pgs.getWidth();
        int y = dst / _pgs.getWidth();
        Pos dstP = new Pos(x, y);
        
        UnitAction astarMove = _astarPath.findPathToAdjacentPosition(src, dst, _gs, fullResourceUse());
        if (astarMove == null || timeRemaining(false) <= 35)
            return astarMove;
        
        int radius = _pgs.getUnits().size() > 32 ? 42 : 64;
        UnitAction ua = findPath(src, dstP, radius);
        if (ua != null)
            return ua;
        return astarMove;
    }

    public Mayari(UnitTypeTable utt) {
        super(-1, -1);
        _utt = utt;
        restartPathFind(); //FloodFillPathFinding(); //AStarPathFinding();
        _memHarvesters = new ArrayList<>();
                
        _dirs = new ArrayList<>();
        _dirs.add(UnitAction.DIRECTION_UP);
        _dirs.add(UnitAction.DIRECTION_DOWN);
        _dirs.add(UnitAction.DIRECTION_LEFT);
        _dirs.add(UnitAction.DIRECTION_RIGHT);
    }
    @Override
    public void reset() {
        _memHarvesters = new ArrayList<>();
        restartPathFind(); //FloodFillPathFinding();//BFSPathFinding();//AStarPathFinding();
    }
    @Override
    public AI clone() {
        return new Mayari(_utt);
    }
    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }

    void printDebug(String str) {
        System.out.println(str);
    }
    
    ResourceUsage fullResourceUse() {
        ResourceUsage ru = _gs.getResourceUsage().clone();
        ru.merge(_pa.getResourceUsage());
        
        //todo - on small board taking future pos as used may 
        //be to harsh and costly
        for (Integer pos : _locationsTaken) {
            int x = pos % _pgs.getWidth();
            int y = pos / _pgs.getWidth();
            Unit u = new Unit(0, _utt.getUnitType("Worker"), x, y);
            UnitAction a = null;
            if (x > 0)
                a = new UnitAction(UnitAction.TYPE_MOVE, NoDirection); //this is a hack
            else
                a = new UnitAction(UnitAction.TYPE_MOVE, NoDirection);
            UnitActionAssignment uaa = new UnitActionAssignment(u, a, 0);
            ru.merge(uaa.action.resourceUsage(u, _pgs));
        }
        return ru;
    }
    boolean outOfBound(Pos p) {
        if(p.getX() < 0 || p.getY() < 0 || p.getX() >= _pgs.getWidth()
                || p.getY() >= _pgs.getHeight())
            return true;
        return false;
    }
    boolean posFree(int x, int y, int dir) {
        Pos pos = futurePos(x, y, dir);
        int rasterPos = pos.getX() + pos.getY() * _pgs.getWidth();
        if(_locationsTaken.contains(rasterPos))
            return false;
        if(_pgs.getUnitAt(pos.getX(), pos.getY()) != null)
            return false;
        if (_pgs.getTerrain(pos.getX(), pos.getY()) == TERRAIN_WALL)
            return false;
        return true;
    }
    void lockPos(int x, int y, int dir) {
        Pos pos = futurePos(x, y, dir);
        int rasterPos = pos.getX() + pos.getY() * _pgs.getWidth();
        _locationsTaken.add(rasterPos);
    }
    Pos futurePos(int x, int y, int dir) {
        int nx = x;
        int ny = y;
        switch (dir) {
            case UnitAction.DIRECTION_DOWN:
                ny = (ny == _pgs.getHeight()- 1) ? ny : ny + 1;
                break;
            case UnitAction.DIRECTION_UP:
                ny = (ny == 0) ? ny : ny - 1;
                break;
            case UnitAction.DIRECTION_RIGHT:
                nx = (nx == _pgs.getWidth() - 1) ? nx : nx + 1;
                break;
            case UnitAction.DIRECTION_LEFT:
                nx = (nx == 0) ? nx : nx - 1;
                break;
            default:
                break;
        }
        return new Pos(nx, ny);
    }
    
    int toDir(Pos src, Pos dst) {
       int dx = dst.getX() - src.getX();
       int dy = dst.getY() - src.getY();
       int dirX = dx > 0 ? UnitAction.DIRECTION_RIGHT : UnitAction.DIRECTION_LEFT;
       int dirY = dy > 0 ? UnitAction.DIRECTION_DOWN : UnitAction.DIRECTION_UP;
       if (Math.abs(dx) > Math.abs(dy))
           return dirX;
       return dirY;
    }
    
    Pos toPos(Unit u) {
        return new Pos(u.getX(), u.getY());
    }
    Pos futurePos(Unit unit) {
        UnitActionAssignment aa = _gs.getActionAssignment(unit);
        if (aa == null)
            return new Pos(unit.getX(), unit.getY());
        if (aa.action.getType() == UnitAction.TYPE_MOVE)
            return futurePos(unit.getX(), unit.getY(), aa.action.getDirection());
        return new Pos(unit.getX(), unit.getY());
    }
    boolean isEnemyUnit(Unit u) {
        return u.getPlayer() >= 0 && u.getPlayer() != _p.getID(); //can be neither ally ot foe
    }
    boolean busy(Unit u) {
        if(_pa.getAction(u) != null)
            return true;
        UnitActionAssignment aa = _gs.getActionAssignment(u);
        return aa != null;
    }
    boolean dying(Unit u) {
        return u.getHitPoints() <= _newDmgs.getOrDefault(u, 0);
    }
    
    boolean willEscapeAttack(Unit attacker, Unit runner) {
        UnitActionAssignment aa = _gs.getActionAssignment(runner);
        if (aa == null)
            return false;
        if (aa.action.getType() != UnitAction.TYPE_MOVE)
            return false;
        int eta = aa.action.ETA(runner) - (_gs.getTime() - aa.time);
        return eta <= attacker.getAttackTime();
    }
    boolean soonInAttackRange(Unit attacker, Unit runner) {
        return squareDist(toPos(attacker), futurePos(runner)) <= attacker.getAttackRange();
    }
    boolean inAttackRange(Unit attacker, Unit runner) {
        return squareDist(toPos(attacker), toPos(runner)) <= attacker.getAttackRange();
    }
    List<Pos> toPos(List<Unit> units) {
        List<Pos> poses = new ArrayList<>();
        for (Unit u : units) {
            poses.add(toPos(u));
        }
        return poses;
    }
    Unit closest(Pos src, List<Unit> units) {
        if (units.isEmpty())
            return null;
        Unit closest = units.stream().min(Comparator.comparing(u -> distance(src, toPos(u)))).get();
        return closest;
    }
    Unit closest(Unit src, List<Unit> units) {
        return closest(toPos(src), units);
    }
    int minDistance(Pos p, List<Pos> poses) {
        int minDist = Integer.MAX_VALUE;
        for (Pos u : poses) {
            minDist = minDist < distance(p, u) ? minDist : distance(p, u);
        }
        return minDist;
    }
    boolean isSeperated(Unit base, List<Unit> units) {
        for (Unit u : units) {
            int rasterPos = u.getX() + u.getY() * _pgs.getWidth();
            ResourceUsage rsu = fullResourceUse();//(_pgs.getHeight() == 8) ? _gs.getResourceUsage() :  //todo - remove this
            if (_astarPath.findPathToAdjacentPosition(base, rasterPos, _gs, rsu) != null)
                    return false;
        }
        return true;
    }
    
    //todo - fix this to real distance
    int distance(Pos a, Pos b) {
        if (a == null | b == null)
            return Integer.MAX_VALUE;
        int dx = a.getX() - b.getX();
        int dy = a.getY() - b.getY();
        return Math.abs(dx) + Math.abs(dy);
    }
    int distance(Unit a, Unit b) {
        return distance(toPos(a), toPos(b));
    }
    int distance(Unit a, Pos b) {
        return distance(toPos(a), b);
    }
    
    double squareDist(Pos p, Pos u) {
        int dx = p.getX() - u.getX();
        int dy = p.getY() - u.getY();
        return Math.sqrt(dx * dx + dy * dy);        
    }
    List<Pos> allPosDist(Pos src, int dist) {
        List<Pos> poss = new ArrayList<>();
        int sx = src.getX();
        int sy = src.getY();
        
        for (int x = -dist; x <= dist; x ++) {
            int y = dist - Math.abs(x);
            poss.add(new Pos(sx + x, sy + y));
            if (y != 0)
                poss.add(new Pos(sx + x, sy - y));
        }
        return poss;
    }
    List<Pos> allPosRange(Pos src, int range) {
         List<Pos> poss = new ArrayList<>();
         for (int r = 0; r <= range; r++)
             poss.addAll(allPosDist(src, r));
         return poss;
    }
    
    boolean attackNow(Unit a, Unit e) {
        UnitAction ua = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, e.getX(), e.getY());
        if (!_gs.isUnitActionAllowed(a, ua))
            return false;
        
        _pa.addUnitAction(a, ua);
        if (!_newDmgs.containsKey(e))
            _newDmgs.put(e, 0);
        int newDmg = _newDmgs.get(e) + a.getMaxHitPoints();
        _newDmgs.replace(e, newDmg);
        return true;
    }
    boolean doNothing(Unit u) {
         _pa.addUnitAction(u, new UnitAction(UnitAction.TYPE_NONE, 1));
         return true;
    }
    boolean moveTowards(Unit a, Pos e) {
        int pos = e.getX() + e.getY() * _pgs.getWidth();
        UnitAction move = findPathAdjacent(a, pos);
        if(move == null)
            return false;
        if (!_gs.isUnitActionAllowed(a, move))
            return false;
        Pos futPos = futurePos(a.getX(), a.getY(), move.getDirection());
        int fPos = futPos.getX() + futPos.getY() * _pgs.getWidth();
        if (_locationsTaken.contains(fPos))
            return false;
        _pa.addUnitAction(a, move);
        _locationsTaken.add(fPos);
        return true;
    }
    boolean safeMoveTowards(Unit a, Unit e) {
        int pos = e.getX() + e.getY() * _pgs.getWidth();
        UnitAction move = findPathAdjacent(a, pos);
        if(move == null)
            return false;
        Pos futurePos = futurePos(a.getX(), a.getY(), move.getDirection());
        int fPos = futurePos.getX() + futurePos.getY() * _pgs.getWidth();
        if(!posFree(futurePos.getX(), futurePos.getY(), NoDirection))
            return false;
        
        for (Unit enemy : _enemies) {
            if (!enemy.getType().canAttack)
                continue;
            int futureDist = distance(futurePos, futurePos(enemy));
            if(futureDist <= enemy.getAttackRange())
                return false;
        }
        
        _pa.addUnitAction(a, move);
        _locationsTaken.add(fPos);
        return true;
    }
    boolean tryMoveAway(Unit a, Unit b) {
        int startDist = distance(toPos(a), toPos(b));
        List<Integer> dirsRand = new ArrayList<>( _dirs ) ;
        Collections.shuffle(dirsRand) ;

        for (int dir : dirsRand) {
            Pos newPos = futurePos(a.getX(), a.getY(), dir);
            if (distance(newPos, toPos(b)) <= startDist)
                continue;
            if (!posFree(newPos.getX(), newPos.getY(), NoDirection)) //a hack
                continue;
            UnitAction ua = new UnitAction(UnitAction.TYPE_MOVE, dir);
            if (_gs.isUnitActionAllowed(b, ua)) {
                _pa.addUnitAction(a, ua);
                lockPos(newPos.getX(), newPos.getY(), NoDirection);
                return true;
            }
        }
        return false;
    }
    boolean moveInDirection(Unit a, Unit b) {
        int startDist = distance(toPos(a), toPos(b));
        List<Integer> dirsRand = new ArrayList<>( _dirs );
        Collections.shuffle(dirsRand);
        for (int dir : dirsRand) {
            Pos newPos = futurePos(a.getX(), a.getY(), dir);
            if (distance(newPos, toPos(b)) >= startDist)
                continue;
            if (!posFree(newPos.getX(), newPos.getY(), NoDirection)) //a hack
                continue;
            UnitAction ua = new UnitAction(UnitAction.TYPE_MOVE, dir);
            if (_gs.isUnitActionAllowed(a, ua)) {
                _pa.addUnitAction(a, ua);
                lockPos(newPos.getX(), newPos.getY(), NoDirection);
                return true;
            }
        }
        return false;
    }
    
    
    boolean produce(Unit u, int dir, UnitType bType) {
        if (busy(u))
            return false;
        if(_p.getResources() - _resourcesUsed < bType.cost)
            return false;
        if(!posFree(u.getX(), u.getY(), dir))
            return false;
        UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, dir, bType);
        if (!_gs.isUnitActionAllowed(u, ua))
            return false;
        _pa.addUnitAction(u, ua);
        lockPos(u.getX(), u.getY(), ua.getDirection());
        if (bType == _utt.getUnitType("Barracks"))
            _futureBarracks.add(futurePos(u.getX(), u.getY(), ua.getDirection()));
        else if (bType == _utt.getUnitType("Heavy"))
            _futureHeavies += 1;
        _resourcesUsed += bType.cost;
        return true;
    }
    boolean produceWherever(Unit u, UnitType bType) {
        for (int dir : _dirs)
            if (produce(u, dir, bType))
                return true;
        return false;
    }
    boolean harvest(Unit worker, Unit resource) {
        if (busy(worker))
            return false;
        if (distance(toPos(worker), toPos(resource)) != 1) {
            System.out.println("wanted to harvest but the resource is not nearby");
            return false;
        }
        int dir = toDir(toPos(worker), toPos(resource));
        UnitAction ua = new UnitAction(UnitAction.TYPE_HARVEST, dir);
        if (!_gs.isUnitActionAllowed(worker, ua))
            return false;
        _pa.addUnitAction(worker, ua);
        return true;
    }
    boolean returnHarvest(Unit worker, Unit base) {
        if (busy(worker))
            return false;
        if (distance(toPos(worker), toPos(base)) != 1) {
            System.out.println("wanted to return but the base is not nearby");
            return false;
        }
        int dir = toDir(toPos(worker), toPos(base));
        UnitAction ua = new UnitAction(UnitAction.TYPE_RETURN, dir);
        if (!_gs.isUnitActionAllowed(worker, ua))
            return false;
        _pa.addUnitAction(worker, ua);
        return true;
    }
    
    boolean overPowering() {
        int power = 0;
        for (Unit u : _allyCombat)
            power += u.getMaxDamage();
        int ePower = 0;
        for (Unit u : _enemiesCombat)
            ePower += u.getMaxDamage();
        return (power - (int) 1.2*ePower) > 0;
    }
    
    int combatScore(Unit u, Unit e) {
        int score = -distance(u, e);
        
        if (u.getType() == _utt.getUnitType("Ranged") 
                && e.getType() == _utt.getUnitType("Ranged") && _pgs.getWidth() > 9)
            score += 2; //todo may be change that and add logic below
        
        if (_pgs.getWidth() >= 16 && (u.getType() == _utt.getUnitType("Heavy") || u.getType() == _utt.getUnitType("Ranged"))
               && (e.getType() == _utt.getUnitType("Barracks"))) //todo - remove? todo base
            score += _pgs.getWidth();
        
        return score;
    }
    
    int[] getCombatScores(Unit u, List<Unit> targets) {
        int[] scores = new int[targets.size()];
        int counter = 0;
        for (Unit t : targets) {
            scores[counter] = combatScore(u, t);
            counter++;
        }
        return scores;
    }
    
    void goCombat(List<Unit> units, int timeToSave) {
        for(Unit u : units) {
            if(busy(u) || !u.getType().canAttack)
                continue;
            
            List<Unit> candidates = new ArrayList(_enemies);
            List<Unit> candidatesCopy = new ArrayList(candidates);
            int[] scores = getCombatScores(u, candidates);
            Collections.sort(candidates, Comparator.comparing(e -> -scores[candidatesCopy.indexOf(e)])); //- for ascending order
            int counter = 0;
            int cutOff = _enemiesCombat.size() > 24 ? 12 : 24; //for performance
            long timeRemain = timeRemaining(true);
            
            while(counter < candidates.size() && counter < cutOff && timeRemain > timeToSave) {
                Unit enemy = candidates.get(counter);
                if (moveTowards(u, futurePos(enemy)))
                    break;
                counter++;
            }
            if (counter < candidates.size()) //if (!candidates.isEmpty()) //did we make a move
                continue;
            if (u.getType() != _utt.getUnitType("Ranged"))
                continue;
            Unit enemy = candidates.get(0);
            if (overPowering()) //give worker to open pathway if blocked
                tryMoveAway(u, u);
            moveInDirection(u, enemy);
        }
    }
    
    boolean shouldWorkersAttack() {
        if (_pgs.getWidth() <= 12)
            return true;
        if (enemyHeaviesWeak() && _enemyArchers.isEmpty() &&
                 _heavies.isEmpty() && _futureHeavies == 0 && _archers.isEmpty())
            return true;
        return false; //todo here
    }
    
    int harvestScore(Unit worker, List<Unit> basesRemain) {
        if (busy(worker) || worker.getResources() > 0)
            return Integer.MAX_VALUE;
        Unit closestResource = closest(worker, _resources);
        Unit closestBase = closest(worker, basesRemain);
        if (closestResource == null || closestBase == null)
            return Integer.MAX_VALUE;
        return distance(toPos(worker), toPos(closestBase)) + distance(toPos(worker), toPos(closestResource));
    }
    boolean goHarvesting(Unit worker) {
        Unit closestRes = closest(worker, _resources);
        if (closestRes == null)
            return false;
        int dist = distance(toPos(worker), toPos(closestRes));
        if (dist == 1) {
            harvest(worker, closestRes); //todo - safe to harvest
            return true;
        }

        if (!moveTowards(worker, toPos(closestRes)))
            tryMoveAway(worker, worker); //random move to shake things up
        return true;
    }
    
    int harvesterPerBase() {
        int totalWorkers = _workers.size() + _enemyWorkers.size();
        int totalCombat = _allyCombat.size() + _enemiesCombat.size();
        int totalResource = _resources.size();
        int baseTotal = _enemyBases.size() + _bases.size();
        int barracks = _barracks.size() + _enemyBarracks.size();
        int area = _pgs.getWidth()*_pgs.getHeight();

        int totalOcc = totalWorkers + totalCombat + baseTotal + barracks + totalResource;
        if(_pgs.getWidth() <= 12 && totalOcc > (int) (area / 2.9))
            return 1; //be more aggresive
        return 2;
    }
    
    void workerAction() {
        List<Unit> ws = new ArrayList<>(_workers);
        List<Unit> bs = new ArrayList<>(_bases);
        
        HashMap<Unit, Integer> baseHarCount = new HashMap<>();
        
        int perBase = harvesterPerBase();
        
        for (Long harId : _memHarvesters) {
            Unit h = _pgs.getUnit(harId);
            Unit b = closest(h, bs);
            if (!busy(h) && h.getResources() == 0)
                goHarvesting(h);
            ws.remove(h);
            if(baseHarCount.containsKey(b))
                baseHarCount.replace(b, baseHarCount.get(b) + 1);
            else
                baseHarCount.put(b, 1);
        }
        
        for (Unit b : baseHarCount.keySet()) {
            if (baseHarCount.get(b) >= perBase) {
                bs.remove(b);
            }
        }
        
        //find harvesters
        while (!bs.isEmpty() && !ws.isEmpty()) {
            Unit w = ws.stream().min(Comparator.comparingInt((e) -> harvestScore(e, bs))).get();
            if (harvestScore(w, bs) == Integer.MAX_VALUE)
                break;
            Unit b = closest(w, bs);
            goHarvesting(w);
            _memHarvesters.add(w.getID());
            ws.remove(w);
            if(baseHarCount.containsKey(b) == false)
                baseHarCount.put(b, 1);
            if(baseHarCount.getOrDefault(b, 0) >= perBase) //top 2 harvesters per base
                bs.remove(b);
        }
        
        for (Unit worker : _workers) {
            if(busy(worker))
                continue;
            if (worker.getResources() <= 0)
                continue;
            Unit base = closest(worker, _bases);
            if (base == null)
                return;
            else if (distance(worker, base) <= 1)
                returnHarvest(worker, base); //todo - check if safe?
            else
                moveTowards(worker, toPos(base));
        }
    }
    
    int bestBuildWorkerDir(Unit base) {
        int bestScore = -Integer.MAX_VALUE;
        int bestDir = 0;
        for (int dir : _dirs) {
            int score = 0;
            Pos n = futurePos(base.getX(), base.getY(), dir);
            if(outOfBound(n) || _pgs.getTerrain(n.getX(), n.getY()) == TERRAIN_WALL)
                continue;
            Unit u = _pgs.getUnitAt(n.getX(), n.getY());
            if (u != null)
                continue;
            if (!posFree(n.getX(), n.getY(), dir))
                continue;
            Unit e = closest(base, _enemies);
            Unit r = closest(base, _resources);
            if (e == null) //already won?
                    continue;
            //towards enemy, or 
            if (r == null ||_workers.size() >= 2*_bases.size()) {// todo here *2?
                score = -distance(n, toPos(e)); //close to enemy is better
            } else
                score = -distance(n, toPos(r)); //close to resource
            if(score > bestScore) {
                bestScore = score;
                bestDir = dir;
            }
        }
        return bestDir;
    }
    
    int workerPerBase(Unit base) {
        if (_pgs.getWidth() < 9)// && _barracks.isEmpty())
            return 15;
        
        if (_pgs.getWidth() > 16)
            return 2;
        
        if (isSeperated(base, _enemies) || _gs.getTime() > 1000)
            return 2;
        
        int enemyFromBelow = (_enemyWorkers.size()) / Math.max(_enemyBases.size(), 1);
        return Math.max(enemyFromBelow, 2);
        //return  .size()
        //return 4;
    }
    
    void basesAction() {
        int producingWorker = 0;
        long producingCount = _bases.stream().filter(b -> _gs.getActionAssignment(b) != null).count();
        for (Unit base : _bases) {
            if(busy(base))
                continue;
            int workerPerBase = workerPerBase(base);
            boolean onlyOption = _resources.isEmpty() && ((_p.getResources() - _resourcesUsed) == 1); //todo some workers carry...
            if(onlyOption) {
                produceWherever(base, _utt.getUnitType("Worker"));
                continue;
            }
            // Dont produce if not in abundance
            if (_pgs.getWidth()>= 9 &&  _workers.size() + producingWorker + producingCount >= workerPerBase * _bases.size())
                continue;
            int dirBuild = bestBuildWorkerDir(base);
            boolean succ = produce(base, dirBuild, _utt.getUnitType("Worker"));
            if (!succ)
                succ = produceWherever(base, _utt.getUnitType("Worker"));
            producingWorker+= succ ? 1 : 0;
        }
    }
    
    boolean produceCombat(Unit barrack, UnitType unitType) {
        List<Integer> dirsLeft = new ArrayList<> (_dirs);
        while(!dirsLeft.isEmpty()) {
            int bestScore = -Integer.MAX_VALUE;
            int bestDir =  -Integer.MAX_VALUE;
            for (Integer dir : dirsLeft) {   
                Pos p = futurePos(barrack.getX(), barrack.getY(), dir);
                int score = -minDistance(p, toPos(_enemies));
                if(score > bestScore) {
                    bestScore = score;
                    bestDir = dir;
                }
            }
            if(produce(barrack, bestDir, unitType))
                return true;
            dirsLeft.remove(Integer.valueOf(bestDir));
        }
        return false;
    }
    boolean enemyHeaviesWeak() {
        if (_enemyFutureHeavy > 0)
            return false;
        if (_enemyHeavies.size() > 1)
            return false;

        if (_enemyHeavies.size() == 1) {
            if(_enemyHeavies.get(0).getHitPoints() > 3) //rangers get 3 shoots at heavy
                return false;
        }
        
        int totEnemyRes = _enemyP.getResources();
        for (Unit u : _enemyWorkers) {
            Pos uPos = new Pos(u.getX(), u.getY());
            int baseDist = minDistance(uPos, toPos(_enemyBases));
            int resDist =  u.getResources() > 0 ? 0 : minDistance(uPos, toPos(_resources));
            
            //todo - here what matters is how close are we to attack relative to future heavies
            totEnemyRes += (baseDist + resDist) < _pgs.getWidth()/2 ? 1 : 0;
        }
        if (totEnemyRes >= _utt.getUnitType("Heavy").cost)
            return false;
        return true;
    }
    void barracksAction() {
        int totalHeavies = _heavies.size() + _futureHeavies;
        for (Unit barrack : _barracks) {
            if (busy(barrack))
                continue;
            
            if(isSeperated(barrack, _enemies)) {
                if(produceCombat(barrack, _utt.getUnitType("Ranged")))
                    continue;
            }
            
            if(produceCombat(barrack, _utt.getUnitType("Heavy")))
                continue;
            
            if(enemyHeaviesWeak()) //not enough resource for heavy
                if(produceCombat(barrack, _utt.getUnitType("Ranged")))
                    continue;
            
            if (_resources.isEmpty() && _p.getResources() - _resourcesUsed < _utt.getUnitType("Heavy").cost)
                produceCombat(barrack, _utt.getUnitType("Ranged"));
        }
    }
    
    boolean validForFutureBuild(Pos p) {
        if(outOfBound(p) || _pgs.getTerrain(p.getX(), p.getY()) == TERRAIN_WALL)
            return false;
        Unit exUnit = _pgs.getUnitAt(p.getX(), p.getY());
        if (exUnit != null && (exUnit.getType() == _utt.getUnitType("Base")
                || exUnit.getType() == _utt.getUnitType("Barracks"))) //todo - may be if mobile unit too?
            return false; 
        return true;
    }
    void buildBase() {
        if (_workers.isEmpty())
            return;
        if(!_bases.isEmpty())
            return;
        if (_resources.isEmpty())
            return;
        if (_p.getResources() - _resourcesUsed < _utt.getUnitType("Base").cost)
            return;
        
        Unit worker = _workers.stream().min(Comparator.comparingInt
        ((e) -> busy(e) ? Integer.MAX_VALUE : minDistance(toPos(e), toPos(_resources)))).get();
        
        if(worker == null || busy(worker))
            return;
        
        for (int dir : _dirs) { //todo get best dir
            Pos p = futurePos(worker.getX(), worker.getY(), dir);
            if (validForFutureBuild(p) && 
                    produce(worker, dir, _utt.getUnitType("Base")))
                return;
        }
    }
    
    boolean between(Pos a, Pos b, Pos c) {
        if (a.getX() < b.getX() && c.getX() < b.getX())
            return false;
        if (a.getX() > b.getX() && c.getX() > b.getX())
            return false;
        if (a.getY() < b.getY() && c.getY() < b.getY())
            return false;
        if (a.getY() > b.getY() && c.getY() > b.getY())
            return false;
        return true;
    }
        
    int buildBarrackWorkerScore(Pos dst, Unit w) {
        if (busy(w))
            return Integer.MIN_VALUE;
        int barrackTLen = _utt.getUnitType("Barracks").produceTime /  _utt.getUnitType("Worker").moveTime;
        int heavyTLen = _utt.getUnitType("Heavy").produceTime  /  _utt.getUnitType("Worker").moveTime;
        int dangerTLen = barrackTLen + heavyTLen;
        Unit e = closest(dst, _enemies);
        int edist = Math.max(distance(dst, toPos(e)), 1);
        int dangerPenalty = 0;
        if (edist < dangerTLen) {
            dangerPenalty = (2*dangerTLen) / edist;
            if (between(toPos(w), dst, toPos(e)))
                dangerPenalty -= 3; //building site is blocking the enemy
        }
        int wDist = distance(toPos(w), dst) / 2;
        return - dangerPenalty - wDist;
    }
    double buildBlockPenalty(Pos p, boolean diagonalsPenalty) {
        double blockingScore = 0;
        List<Pos> nn = allPosRange(p, 2);
        for (Pos n : nn) {
            int dist = distance(n, p);
            if ((!diagonalsPenalty && dist == 2) || dist == 0)
                continue;
            if (outOfBound(n) || _pgs.getTerrain(n.getX(), n.getY()) == TERRAIN_WALL)
                blockingScore += dist > 1 ? 0 : 0.2;
            Unit u = _pgs.getUnitAt(n.getX(), n.getY());
            if(u == null)
                continue;
            if(u.getType().isResource || u.getType() == _utt.getUnitType("Base"))
                blockingScore += dist > 1 ? 1 : 4;
        }
        return blockingScore;
    }
    int buildBarrackScore(Pos dst) {
        if (!validForFutureBuild(dst))
            return Integer.MIN_VALUE;
        if (_workers.isEmpty())
            return Integer.MIN_VALUE;
        
        Unit b = closest(dst, _bases);
        if (isSeperated(b, _enemies))
            return -(int) buildBlockPenalty(dst, true)*10;
        
        List<Pos> allBrxs = toPos(_barracks);
        allBrxs.addAll(_futureBarracks);
        int deseretScore = 0;
        if (!allBrxs.isEmpty())
            deseretScore = (int) (minDistance(toPos(b), allBrxs) / 2); //like base to be deserted
        
        double blockingPenalty = buildBlockPenalty(dst, false);

        Unit worker = _workers.stream().max(Comparator.
                comparingInt((u) -> buildBarrackWorkerScore(dst, u))).get();
        int workerScore = buildBarrackWorkerScore(dst, worker); //include danger
        
        
        return 10*(deseretScore - (int)blockingPenalty + workerScore);
    }
    
    boolean goBuildBarrack(Unit worker, Pos dst) {
        if (distance(toPos(worker), dst) != 1)
            return moveTowards(worker, dst);
        int dir = toDir(toPos(worker), dst);
        return produce(worker, dir,  _utt.getUnitType("Barracks"));
    }
    boolean needNewBarracks() {
        if (_barracks.size() + _futureBarracks.size() >= _bases.size()) //todo
            return false;
        
        int maxDist = _pgs.getWidth() / 4;
        for (Unit b : _bases) {
            int minDist = minDistance(toPos(b), _futureBarracks);
            if (minDist <= maxDist)
                continue;
            Unit brx = closest(b, _barracks);
            if (brx != null && distance(toPos(brx), toPos(b)) < maxDist)
                continue;
            return true;
        }
        return false;
    }
    void buildBracks() {
        if (_p.getResources() - _resourcesUsed  - 1 < _utt.getUnitType("Barracks").cost)
            return;
        
        if (!needNewBarracks())
            return;
        
        if(_bases.isEmpty()) //todo
            return;
        
        if (_workers.isEmpty())
            return;
        
        List<Pos> pCandidates = new ArrayList<>();
        for (Unit base : _bases) {
            List<Pos> poses = allPosRange(toPos(base), 2);
            pCandidates.addAll(poses);
        }
        
        int counter = 0; 
        while (!pCandidates.isEmpty() && counter < 2) {//sometimes better to wait to next round...
            Pos c = pCandidates.stream().max(Comparator.comparingInt((e) -> buildBarrackScore(e))).get();
            if (buildBarrackScore(c) == Integer.MIN_VALUE)
                break;
            Unit worker = _workers.stream().max(Comparator.comparingInt((k) -> buildBarrackWorkerScore(c, k))).get();
            if (buildBarrackWorkerScore(c, worker) == Integer.MIN_VALUE)
                break;
            if(goBuildBarrack(worker, c))
                return;
            pCandidates.remove(c);
            counter += 1;
        }
    }
    
    int combatNearbyScore(Unit attacker, Unit defender) {
        if(dying(defender))
            return Integer.MIN_VALUE;
        
        if (squareDist(toPos(attacker), toPos(defender))
                > (_utt.getUnitType("Ranged").attackRange + 3))
            return Integer.MIN_VALUE; //tood - remove
        
        //int rangerBasePenalty = 0;
        //if (attacker.getType() == _utt.getUnitType("Ranger")
         //       && attacker.getType() == _utt.getUnitType("Base")) 
         //   rangerBasePenalty = 1;
        
        boolean inRange = inAttackRange(attacker, defender);
        int attackSucc = inRange && !willEscapeAttack(attacker, defender) ? 1 : 0;
        int threatened = inAttackRange(defender, attacker) ? 1 : 0;
        int willKill = attacker.getMaxDamage() > defender.getHitPoints() ? 1 : 0;
        
        int enemyPower = defender.getMaxDamage(); //defender.getMaxHitPoints();
        
        int archerToWorker = (attacker.getType() == _utt.getUnitType("Ranged") && //todo big change this was Ranger instead of Ranged
                defender.getType() == _utt.getUnitType("Worker")) ? 1 : 0;  //to do this was 1: 0 
        
        return 1000*attackSucc + 100 * willKill + (archerToWorker + enemyPower) * 10 + threatened;
    }
    boolean attackNearby(Unit u, Unit e) {
        boolean inRange = inAttackRange(u, e);
        boolean attackSucc = inRange && !willEscapeAttack(u, e);
        if (attackSucc) {
            return attackNow(u, e);
        }
        else if(soonInAttackRange(u, e)) { //wait for attack
            doNothing(u);
            return true;
        }
        boolean threatened = inAttackRange(e, u);
        if(threatened) {
            return moveTowards(u, toPos(e)); //running to rangers... may be shouldnt?
        }
        return false;
    }
    void attackNearby(Unit u) {
        List<Unit> candidates = new ArrayList<>(_enemies);
        
        int cutOff = _enemiesCombat.size() > 24 ? 12 : 24; //for performance issue
        int counter = 0;
        while (!candidates.isEmpty() && counter < cutOff) {
            Unit c = candidates.stream().max(Comparator.comparingInt((e) -> combatNearbyScore(u, e))).get();
            if (attackNearby(u, c))
                break;
            candidates.remove(c);
            counter++;
        }
    }
    void attackNearby() {
        for (Unit u : _allyUnits) {
            if (busy(u) || !u.getType().canAttack)
                continue;
            attackNearby(u);
        }
    }
    
    void initTimeLimit() {
        _startCycleMilli = 0;
        _latestTsMilli = 0;
        
        //kinda random, do not want to take time unnecessarily
        if (_pgs.getWidth() < 24 || _pgs.getUnits().size() < 24)
            return;
        
        _startCycleMilli = System.currentTimeMillis();
        _latestTsMilli = _startCycleMilli;
    }    
    long timeRemaining(boolean updateTs) {
        int perCycleTime = 100;
        if (_startCycleMilli == 0)
            return perCycleTime;
        
        if (updateTs)
            _latestTsMilli = System.currentTimeMillis();
        
        return perCycleTime - (_latestTsMilli - _startCycleMilli);
    }
    
    void init() {
        _resourcesUsed = 0;
        _locationsTaken = new ArrayList<>();
        
        _resources = new ArrayList<>();
        _all = new ArrayList<>(); 
        
        _allyCombat = new ArrayList<>(); 
        _allyUnits = new ArrayList<>(); 
        
        _bases = new ArrayList<>();
        _barracks = new ArrayList<>();
        _workers = new ArrayList<>();
        _heavies = new ArrayList<>();
        _archers = new ArrayList<>();
        _lights = new ArrayList<>();

        _enemies  = new ArrayList<>();
        _enemiesCombat = new ArrayList<>();
        
        _enemyBases = new ArrayList<>();
        _enemyBarracks = new ArrayList<>();
        _enemyWorkers = new ArrayList<>();
        _enemyHeavies = new ArrayList<>();
        _enemyArchers = new ArrayList<>();
        _enemyLights = new ArrayList<>();
        
        
        
        _newDmgs = new HashMap<>();
        
        _dirs = new ArrayList<>();
        _dirs.add(UnitAction.DIRECTION_UP);
        _dirs.add(UnitAction.DIRECTION_DOWN);
        _dirs.add(UnitAction.DIRECTION_LEFT);
        _dirs.add(UnitAction.DIRECTION_RIGHT);


        
        for (Unit u : _pgs.getUnits()) {
            if (u.getType().isResource)
                _resources.add(u);
            else if (u.getType() == _utt.getUnitType("Base") && isEnemyUnit(u))
                _enemyBases.add(u);
            else if (u.getType() == _utt.getUnitType("Base"))
                _bases.add(u);
            else if (u.getType() == _utt.getUnitType("Barracks") && isEnemyUnit(u))
                _enemyBarracks.add(u);
            else if (u.getType() == _utt.getUnitType("Barracks"))
                _barracks.add(u);
            else if (u.getType() == _utt.getUnitType("Worker") && isEnemyUnit(u))
                _enemyWorkers.add(u);
            else if (u.getType() == _utt.getUnitType("Worker"))
                _workers.add(u);
            else if (u.getType() == _utt.getUnitType("Ranged") && isEnemyUnit(u))
                _enemyArchers.add(u);
            else if (u.getType() == _utt.getUnitType("Ranged"))
                _archers.add(u);
            else if (u.getType() == _utt.getUnitType("Heavy") && isEnemyUnit(u))
                _enemyHeavies.add(u);
            else if (u.getType() == _utt.getUnitType("Heavy"))
                _heavies.add(u);
            else if (u.getType() == _utt.getUnitType("Light") && isEnemyUnit(u))
                _enemyLights.add(u);
            else if (u.getType() == _utt.getUnitType("Light"))
                _lights.add(u);     
        }
        for (Unit u : _pgs.getUnits()) {
            if(u.getType().isResource)
                continue;
            _all.add(u);
            if (isEnemyUnit(u))
                _enemies.add(u);
            else
                _allyUnits.add(u);
            if(isEnemyUnit(u) && u.getType().canAttack)
                _enemiesCombat.add(u);
            else if(u.getType().canAttack)
                _allyCombat.add(u);
        }
        
        _futureBarracks = new ArrayList<>();
        _futureHeavies = 0;
        _enemyFutureHeavy = 0;
        for (Unit u : _all) { //todo big change that was ally by mistake
            UnitActionAssignment aa = _gs.getActionAssignment(u);
            if(aa == null)
                continue;
            if (aa.action.getType() != UnitAction.TYPE_PRODUCE)
                 continue;
             
            lockPos(u.getX(), u.getY(), aa.action.getDirection());
             
            UnitType ut = aa.action.getUnitType();
            
            if (!isEnemyUnit(u) && (ut != null))
                 _resourcesUsed += aa.action.getUnitType().cost; 
            
             if (!isEnemyUnit(u) && ut == _utt.getUnitType("Barracks")) {
                 Pos p = futurePos(u.getX(), u.getY(), aa.action.getDirection()); //todo this was aa.action.x which was 0 big change
                 _futureBarracks.add(p);
             }
             if (!isEnemyUnit(u) && ut == _utt.getUnitType("Heavy")) {
                 _futureHeavies += 1;
             }
             if(isEnemyUnit(u) && aa.action.getUnitType() == _utt.getUnitType("Heavy"))
                 _enemyFutureHeavy += 1;
        }
        
        for (Unit u : _all) {
            UnitActionAssignment aa = _gs.getActionAssignment(u);
            if(aa == null)
                continue;
             if (aa.action.getType() != UnitAction.TYPE_ATTACK_LOCATION)
                 continue;
            Unit t = _pgs.getUnitAt( aa.action.getLocationX(), aa.action.getLocationY());
            if (t == null)
                continue;
            if (!_newDmgs.containsKey(t))
                _newDmgs.put(t, 0);
            // todo - not assuming its going to hit
            int newDmg = _newDmgs.get(t) + u.getMaxDamage();
            _newDmgs.replace(t, newDmg);
        }
        
        if (_bases.size() == 0) 
            _memHarvesters.clear();
        
        Iterator<Long> iterH = _memHarvesters.iterator();
        while (iterH.hasNext()) {
          Long id = iterH.next();
          if (_pgs.getUnit(id) == null) iterH.remove();
        }
        
        initTimeLimit();
    }
    
    void freeBlocks(List<Unit> units) {
        for (Unit u : units) {
            if (busy(u))
                continue;
            
            List<Pos> poses = allPosDist(toPos(u), 1);
            boolean somebodyNear = false;
            for (Pos p : poses) {
                if(_pgs.getUnitAt(p.getX(), p.getY())!=null) {
                    somebodyNear = true;
                    break;
                }
            }
            if (somebodyNear)
                tryMoveAway(u, u);
        }
    }
    
    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        _gs = gs;
        _pgs = gs.getPhysicalGameState();
         _p = gs.getPlayer(player);
        _enemyP = gs.getPlayer(player == 0 ? 1 : 0);
        
        _pa = new PlayerAction();
        
        init();
        
        attackNearby(); //fight whoever is near
        
        
        buildBracks();
        buildBase();
        barracksAction();
        basesAction();
        
        workerAction();
        
        if (shouldWorkersAttack())
            goCombat(_workers, 35);
        else
            freeBlocks(_workers);
        
        goCombat(_heavies, 30);
        goCombat(_archers, 15);
        goCombat(_lights, 5);
        
        //if (_pgs.getWidth() >= 9)
            
        
        
        _pa.fillWithNones(gs, player, 1);
        return _pa;
    }
}