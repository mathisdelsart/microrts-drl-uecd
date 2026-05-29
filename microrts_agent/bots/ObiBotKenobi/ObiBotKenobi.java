import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import util.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class ObiBotKenobi extends AIWithComputationBudget {

    UnitTypeTable m_utt;
    HashMap<Long,Integer> jobs;
    public static final int TYPE_SOLDIER=0;
    public static final int TYPE_HARVESTER=1;
    public static final int TYPE_BUILDER=2;
    private int[][] distances;

    // This is the default constructor that microRTS will call:

    public ObiBotKenobi(UnitTypeTable utt) {

        super(-1,-1);

        m_utt = utt;
        jobs= new HashMap<>();

    }

    // This will be called by microRTS when it wants to create new instances of this bot (e.g., to play multiple games).

    public AI clone() {

        return new ObiBotKenobi(m_utt);

    }



    // This will be called once at the beginning of each new game:

    public void reset() {

    }



    // Called by microRTS at each game cycle.

    // Returns the action the bot wants to execute.

    public PlayerAction getAction(int player, GameState gs) {

        PlayerAction pa = new PlayerAction();
        for (Unit u : gs.getUnits()) {
            if (gs.getActionAssignment(u) != null || u.getPlayer() != player) {
                continue;
            }
            ArrayList<Pair<UnitAction, Integer>> rewards = new ArrayList<>();
            Iterator<UnitAction> it2 = u.getUnitActions(gs).iterator();
            UnitAction a = null;

            while (it2.hasNext()) {
                UnitAction ua = it2.next();
                rewards.add(new Pair<>(ua, evaluate(ua, u, gs, player)));
            }
            while (checkIfUsed(pa, a, u, gs, player)) {
                Pair<UnitAction, Integer> ua = getActionFromReward(rewards);
                a = ua.m_a;
                rewards.remove(ua);
            }
            pa.addUnitAction(u, a);
        }

        return pa;

    }



    // This will be called by the microRTS GUI to get the

    // list of parameters that this bot wants exposed

    // in the GUI.

    public List<ParameterSpecification> getParameters()

    {

        return new ArrayList<>();

    }

    public boolean checkIfUsed(PlayerAction pa, UnitAction a, Unit u,GameState gs,int player){
        if(a==null){
            return true;
        }
        if(a.getDirection()<0||a.getDirection()>=4){
            return false;
        }
        Iterator<Pair<Unit,UnitAction>> it=pa.getActions().iterator();
        int posx1=u.getX()+UnitAction.DIRECTION_OFFSET_X[a.getDirection()];
        int posy1=u.getY()+UnitAction.DIRECTION_OFFSET_Y[a.getDirection()];
        while(it.hasNext()){
            Pair<Unit,UnitAction> uua=it.next();
            if(uua.m_b.getDirection()<0||uua.m_b.getDirection()>=4){
                continue;
            }
            int posx2=uua.m_a.getX()+UnitAction.DIRECTION_OFFSET_X[uua.m_b.getDirection()];
            int posy2=uua.m_a.getY()+UnitAction.DIRECTION_OFFSET_Y[uua.m_b.getDirection()];
            if(posx1==posx2 && posy1==posy2){
                return true;
            }
        }
        for (Unit u2 : gs.getUnits()) {
            if (u2.getPlayer() != player || gs.getActionAssignment(u2) == null) {
                continue;
            }
            UnitAction a2 = gs.getActionAssignment(u2).action;
            if (a2.getDirection() < 0 || a2.getDirection() >= 4) {
                continue;
            }
            int posx2 = u2.getX() + UnitAction.DIRECTION_OFFSET_X[a2.getDirection()];
            int posy2 = u2.getY() + UnitAction.DIRECTION_OFFSET_Y[a2.getDirection()];
            if (posx1 == posx2 && posy1 == posy2) {
                return true;
            }

            if (a2.getType() == UnitAction.TYPE_PRODUCE && a.getType() == UnitAction.TYPE_PRODUCE) {
                if(u.getType().name=="Barracks" && u2.getType().name=="Base") {
                    gs.getActionAssignment(u2).action = new UnitAction(UnitAction.TYPE_NONE);
                    return false;
                }
                if ((gs.getPhysicalGameState().getPlayer(player).getResources() - a2.getUnitType().cost) < a.getUnitType().cost) {
                    return true;
                }
            }
        }



        return false;
    }

    public Pair<UnitAction,Integer> getActionFromReward(ArrayList<Pair<UnitAction,Integer>> rewards){
        Iterator<Pair<UnitAction,Integer>> it=rewards.iterator();
        Pair<UnitAction,Integer> a=null;
        Random r=new Random();
        while(it.hasNext()){
            Pair<UnitAction,Integer> ua= it.next();
            if(a==null){
                a=ua;
                continue;
            }
            if(a.m_b<ua.m_b){
                a=ua;
            }
            if(a.m_b.equals(ua.m_b)){
                double p=r.nextDouble();
                if(p<0.5){
                    a=ua;
                }
            }
        }
        return a;
    }

    public int evaluate(UnitAction a, Unit u, GameState gs, int player){
        if(u.getType().name.equals("Base")){
            return evaluateBase(a,u,gs,player);
        }else if(u.getType().canAttack && !u.getType().canHarvest){
            return evaluateSoldier(a,u,gs,player);
        }else if(u.getType().name.equals("Barracks")){
            return evaluateBarracks(a,u,gs,player);
        }else if(u.getType().name.equals("Worker")){
            if(!jobs.containsKey(u.getID())){
                getJob(u);
            }
            if(jobs.get(u.getID())==TYPE_SOLDIER){
                return evaluateSoldier(a,u,gs,player);
            }
            if(jobs.get(u.getID())==TYPE_HARVESTER){
                return evaluateHarvester(a,u,gs,player);
            }
            if(jobs.get(u.getID())==TYPE_BUILDER){
                return evaluateBuilder(a,u,gs,player);
            }
        }
        return -(gs.getPhysicalGameState().getWidth()*gs.getPhysicalGameState().getHeight());
    }

    public void getJob(Unit u){
        if(!jobs.containsValue(TYPE_HARVESTER)){
            jobs.put(u.getID(),TYPE_HARVESTER);
            return;
        }
        if(!jobs.containsValue(TYPE_BUILDER)){
            jobs.put(u.getID(),TYPE_BUILDER);
            return;
        }
        jobs.put(u.getID(),TYPE_SOLDIER);
    }

    public int evaluateBuilder(UnitAction a,Unit u, GameState gs,int player){
        PhysicalGameState pgs=gs.getPhysicalGameState();
        if(a.getType()==UnitAction.TYPE_ATTACK_LOCATION){
            return 2;
        }
        if(a.getType()==UnitAction.TYPE_NONE){
            return -(pgs.getWidth()*pgs.getHeight());
        }
        int posx=u.getX()+UnitAction.DIRECTION_OFFSET_X[a.getDirection()];
        int posy=u.getY()+UnitAction.DIRECTION_OFFSET_Y[a.getDirection()];
        int pos=posx+posy*pgs.getWidth();
        if(gs.getUnits().stream().noneMatch(Item-> Item.getType().name.equals("Barracks"))){
            if(a.getType()==UnitAction.TYPE_PRODUCE) {
                return -pgs.getUnitsAround(posx,posy , 1).size()+8;
            }
        }
        if(u.getResources()>0){
            if(a.getType()==UnitAction.TYPE_RETURN){
                return 1;
            }
            int distance=getNearestUnit(pos,gs,"Base",player);
            return -distance;
        }
        if(a.getType()==UnitAction.TYPE_HARVEST){
            return 1;
        }
        int distance=getNearestUnit(pos,gs,"Resource",-1);
        return -distance;
    }

    public int evaluateHarvester(UnitAction a,Unit u,GameState gs, int player){
        PhysicalGameState pgs=gs.getPhysicalGameState();
        if(a.getType()==UnitAction.TYPE_ATTACK_LOCATION){
            return 2;
        }
        if(a.getType()==UnitAction.TYPE_NONE||a.getType()==UnitAction.TYPE_PRODUCE){
            return -(pgs.getWidth()*pgs.getHeight());
        }
        int posx=u.getX()+UnitAction.DIRECTION_OFFSET_X[a.getDirection()];
        int posy=u.getY()+UnitAction.DIRECTION_OFFSET_Y[a.getDirection()];
        int pos=posx+posy*pgs.getWidth();
        if(u.getResources()>0){
            if(a.getType()==UnitAction.TYPE_RETURN){
                return 1;
            }
            int distance=getNearestUnit(pos,gs,"Base",player);
            return -distance;
        }
        if(a.getType()==UnitAction.TYPE_HARVEST){
            return 1;
        }
        int distance=getNearestUnit(pos,gs,"Resource",-1);
        return -distance;
    }

    public int evaluateBarracks(UnitAction a,Unit u,GameState gs,int player){
        PhysicalGameState pgs=gs.getPhysicalGameState();
        if(a.getType()==UnitAction.TYPE_NONE|| a.getUnitType().name.equals("Heavy")){
            return -(pgs.getWidth()*pgs.getHeight());
        }
        int posx=u.getX()+UnitAction.DIRECTION_OFFSET_X[a.getDirection()];
        int posy=u.getY()+UnitAction.DIRECTION_OFFSET_Y[a.getDirection()];
        int pos=posx+posy*pgs.getWidth();
        int distance=getNearestUnit(pos,gs,1-player);
        //variante für verschiedene Einheiten einbauen
        int extra=0;
        if(distance>(pgs.getWidth()+pgs.getHeight()/4) && a.getUnitType().name.equals("Ranged")){
            extra=1;
        }
        return -distance+extra;
    }

    public int evaluateSoldier(UnitAction a,Unit u, GameState gs, int player){
        PhysicalGameState pgs=gs.getPhysicalGameState();
        if(a.getType()==UnitAction.TYPE_NONE||a.getType()==UnitAction.TYPE_PRODUCE){
            return -(pgs.getWidth()*pgs.getHeight());
        }
        int pos=u.getPosition(pgs);
        if(a.getType()==UnitAction.TYPE_ATTACK_LOCATION){
            return 1;
        }
        int posx=pos%pgs.getWidth()+UnitAction.DIRECTION_OFFSET_X[a.getDirection()];
        int posy=pos/pgs.getWidth()+UnitAction.DIRECTION_OFFSET_Y[a.getDirection()];
        int distance=getNearestUnit(posx+posy*pgs.getWidth(),gs,1-player);
        return -distance;
    }

    public int evaluateBase(UnitAction a, Unit u, GameState gs, int player){
        PhysicalGameState pgs=gs.getPhysicalGameState();
        if(a.getType()==UnitAction.TYPE_NONE){
            return -(pgs.getWidth()*pgs.getHeight());
        }
        int posx=u.getX()+UnitAction.DIRECTION_OFFSET_X[a.getDirection()];
        int posy=u.getY()+UnitAction.DIRECTION_OFFSET_Y[a.getDirection()];
        int pos=posx+posy*pgs.getWidth();
        if(!jobs.containsValue(TYPE_HARVESTER)){
            int distance=getNearestUnit(pos,gs,"Resource",-1);
            return -distance;
        }
        if(!jobs.containsValue(TYPE_BUILDER)){
            return pgs.getUnitsAround(posx,posy,2).size();
        }
        int distance=getNearestUnit(pos,gs,1-player);
        return -distance;
    }

    public int getNearestUnit(int startPos,GameState gs, String unitName, int player){
        List<Unit> ul=gs.getUnits();
        List<Unit> us=ul.stream().filter(item->item.getType().name.equals(unitName) && item.getPlayer()==player).collect(Collectors.toList());
        PhysicalGameState pgs=gs.getPhysicalGameState();
        ArrayList<Integer> targetPos=new ArrayList<>();
        for(Unit unit:us){
            targetPos.add(unit.getPosition(pgs));
        }
        ArrayList<Pair<Integer,Integer>> distances=djikstra(startPos,targetPos,gs);
        if(distances==null || distances.size()==0){
            return pgs.getWidth()*pgs.getHeight()+1;
        }
        distances.sort(Comparator.comparingInt(a -> a.m_b));
        return distances.get(0).m_b;
    }

    public int getNearestUnit(int startPos,GameState gs, int player){
        List<Unit> ul=gs.getUnits();
        List<Unit> us=ul.stream().filter(item->item.getPlayer()==player).collect(Collectors.toList());
        PhysicalGameState pgs=gs.getPhysicalGameState();
        ArrayList<Integer> targetPos=new ArrayList<>();
        for(Unit unit:us){
            targetPos.add(unit.getPosition(pgs));
        }
        ArrayList<Pair<Integer,Integer>> distances=djikstra(startPos,targetPos,gs);
        if(distances==null || distances.size()==0){
            return pgs.getWidth()*pgs.getHeight()+1;
        }
        distances.sort(Comparator.comparingInt(a -> a.m_b));
        return distances.get(0).m_b;
    }

    public ArrayList<Pair<Integer,Integer>> djikstra(int startPos,ArrayList<Integer> targetPos,GameState gs){
        PhysicalGameState pgs=gs.getPhysicalGameState();
        int w=pgs.getWidth();
        int h=pgs.getHeight();
        int startx=startPos%w;
        int starty=startPos/w;
        ArrayList<Pair<Integer,Integer>> returnDistances=new ArrayList<>();
        if(targetPos.size()==0){
            return null;
        }
        PriorityQueue<Pair<Integer,Integer>> fields= new PriorityQueue<>(Comparator.comparingInt(a -> a.m_b));
        /**
        if(distances==null) {
            distances = new int[w][h];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    distances[j][i] = w * h + 1;
                }
            }
        }
         **/
        if(distances==null){
            distances=new int[w][h];
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if(distances[j][i]<w*h+1){
                    distances[j][i]=w*h+1;
                }
            }
        }

        distances[startx][starty]=0;
        fields.add(new Pair<>(startPos,0));
        ArrayList<Integer> closedList=new ArrayList<>();
        while(!fields.isEmpty() && !targetPos.isEmpty()){
            int pos=fields.remove().m_a;
            int[] neighbors=neighbors(pos,w,h,gs,targetPos);
            for (int neighbor : neighbors) {
                if (neighbor == -1) {
                    continue;
                }
                distances[neighbor % w][neighbor / w] = getDistance(distances[neighbor % w][neighbor / w], distances[pos % w][pos / w]);
                if(!closedList.contains(neighbor) && !fields.contains(new Pair<>(neighbor, distances[neighbor % w][neighbor / w]))) {
                    fields.add(new Pair<>(neighbor, distances[neighbor % w][neighbor / w]));
                    closedList.add(neighbor);
                }
                if(targetPos.remove((Integer) neighbor)){
                    returnDistances.add(new Pair<>(neighbor,distances[neighbor % w][neighbor / w]));
                }
            }
        }
        return returnDistances;
    }

    public int getDistance(int oldDistance,int posDistance){
        return Math.min(posDistance + 1, oldDistance);
    }
    /**
    public int getSmallest(ArrayList<Integer> l,int[][] abstand,  int w){
        int pos=l.get(0);
        for (Integer integer : l) {
            if (abstand[integer % w][integer / w] < abstand[pos % w][pos / w]) {
                pos = integer;
            }
        }

        return pos;
    }
     **/


    public int[] neighbors(int pos,int w,int h,GameState gs, ArrayList<Integer> targetPos){
        int[] neighbors=new int[4];
        int posx=pos%w;
        int posy=pos/w;
        int rightx=posx+1;
        int righty=posy;
        PhysicalGameState pgs=gs.getPhysicalGameState();
        if(!(rightx<w)||pgs.getTerrain(rightx,righty)==1||(pgs.getUnitAt(rightx,righty)!=null && !targetPos.contains(rightx+righty*w))){
            neighbors[0]=-1;
        }else{
            neighbors[0]=rightx+righty*w;
        }

        int leftx=posx-1;
        int lefty=posy;
        if(!(leftx>=0)||pgs.getTerrain(leftx,lefty)==1||(pgs.getUnitAt(leftx,lefty)!=null && !targetPos.contains(leftx+lefty*w))){
            neighbors[1]=-1;
        }else{
            neighbors[1]=leftx+lefty*w;
        }

        int upx=posx;
        int upy=posy-1;
        if(!(upy>=0)||pgs.getTerrain(upx,upy)==1||(pgs.getUnitAt(upx,upy)!=null && !targetPos.contains(upx+upy*w))){
            neighbors[2]=-1;
        }else{
            neighbors[2]=upx+upy*w;
        }

        int downx=posx;
        int downy=posy+1;
        if(!(downy<h)||pgs.getTerrain(downx,downy)==1||(pgs.getUnitAt(downx,downy)!=null && !targetPos.contains(downx+downy*w))){
            neighbors[3]=-1;
        }else{
            neighbors[3]=downx+downy*w;
        }

        return neighbors;

    }

}

