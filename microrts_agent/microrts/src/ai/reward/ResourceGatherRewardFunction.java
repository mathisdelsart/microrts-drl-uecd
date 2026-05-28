package ai.reward;

import rts.GameState;
import rts.PhysicalGameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import util.Pair;

/**
 * Resource gathering reward: +1 per harvest action, +1 per return action.
 *
 * Also sets done=true when no Resource tiles with resources remain on the map.
 *
 * @author costa
 */
public class ResourceGatherRewardFunction extends RewardFunctionInterface
{
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs)
    {
        reward = 0.0;
        for (Pair<Unit, UnitAction> p : te.getActions())
        {
            if (p.m_a.getPlayer() == maxplayer && p.m_b.getType() == UnitAction.TYPE_HARVEST)
            {
                reward += 1;
            }
            else if (p.m_a.getPlayer() == maxplayer && p.m_b.getType() == UnitAction.TYPE_RETURN)
            {
                reward += 1;
            }
        }
        done = true;
        PhysicalGameState pgs = afterGs.getPhysicalGameState();
        for (Unit u : pgs.getUnits())
        {
            if (u.getType().name.equals("Resource") && u.getResources() > 0)
            {
                done = false;
                return;
            }
        }
    }
}
