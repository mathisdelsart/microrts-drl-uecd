package ai.reward;

import rts.GameState;
import rts.TraceEntry;
import rts.UnitAction;
import rts.units.Unit;
import util.Pair;

/**
 * Base production reward: +1 per Base produced.
 *
 * @author costa & mathisdelsart
 */
public class ProduceBaseRewardFunction extends RewardFunctionInterface
{
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs)
    {
        reward = 0.0;
        done = false;
        for (Pair<Unit, UnitAction> p : te.getActions())
        {
            if (p.m_a.getPlayer() == maxplayer && p.m_b.getType() == UnitAction.TYPE_PRODUCE
                    && p.m_b.getUnitType() != null)
            {
                if (p.m_b.getUnitType().name.equals("Base")) { reward += 1; }
            }
        }
    }
}
