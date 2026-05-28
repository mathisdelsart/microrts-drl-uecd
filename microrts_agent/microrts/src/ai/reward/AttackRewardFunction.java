package ai.reward;

import rts.GameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import util.Pair;

/**
 * Attack reward: +1 per attack action on an enemy unit.
 *
 * Friendly fire is impossible in MicroRTS (action masking prevents it),
 * so only opponent targets are checked.
 *
 * @author costa & mathisdelsart
 */
public class AttackRewardFunction extends RewardFunctionInterface
{
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs)
    {
        reward = 0.0;
        done = false;
        for (Pair<Unit, UnitAction> p : te.getActions())
        {
            if (p.m_a.getPlayer() == maxplayer && p.m_b.getType() == UnitAction.TYPE_ATTACK_LOCATION)
            {
                Unit other = te.getPhysicalGameState().getUnitAt(p.m_b.getLocationX(), p.m_b.getLocationY());
                if (other != null && other.getPlayer() == minplayer) { reward += 1; }
            }
        }
    }
}
